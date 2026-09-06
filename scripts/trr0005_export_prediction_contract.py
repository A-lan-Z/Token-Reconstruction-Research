#!/usr/bin/env python3
"""Export TRR-0005 prediction metadata into the frozen row contract.

The first TRR-0005 prediction run wrote the correct prediction tensors and
timing values, but its merged JSON rows retained the legacy receipt schema.
This utility is a serialization-only repair for that one unambiguous defect.
It reads public JSON metadata and opaque prediction bytes, copies each binary
artifact byte-for-byte, rewrites only the row schema and output path
cross-references, and rebuilds the two entry manifests.  The original
``run_evidence.json`` remains in the source tree and is referenced as the
historical execution record; it is never rewritten or copied over.

No source text, truth labels, future activations, model state, or tensor
contents are loaded by this exporter.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from token_reconstruction.trr0005_contract import (
    EXPECTED_CELL_IDS,
    METHOD_IDS,
    PREDICTION_SCHEMA,
    TASK_ID,
    validate_prediction_descriptor,
)


EXPORT_SCHEMA = "token-reconstruction.trr0005-prediction-contract-export.v1"
LEGACY_RECEIPT_SCHEMA = "token-reconstruction.trr0005-prediction-receipt.v1"
PREDICTION_MANIFEST_SCHEMA = "token-reconstruction.trr0005-prediction-descriptor-manifest.v1"
TIMING_MANIFEST_SCHEMA = "token-reconstruction.trr0005-timing-descriptor-manifest.v1"
RUN_EVIDENCE_SCHEMA = "token-reconstruction.trr0005-fresh-confirmation-run.v1"
PREDICTION_MAP_KEY = "predictions"
TIMING_MAP_KEY = "timings"
ALLOWED_DESCRIPTOR_CHANGES = frozenset(
    {
        "schema",
        "prediction_artifact.path",
        "artifact_relative_to_root",
    }
)


class ExportError(RuntimeError):
    """Raised when the metadata-only export cannot prove its invariants."""


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"{description} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"{description} must be a JSON object: {path}")
    return value


def _json_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ExportError(f"refusing to overwrite export file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
    except OSError as exc:
        raise ExportError(f"unable to write export file: {path}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ExportError(f"unable to hash file: {path}") from exc
    return digest.hexdigest()


def _repo_relative(path: Path, *, repository_root: Path) -> str:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ExportError(f"path is outside repository root: {path}") from exc


def _file_record(path: Path, *, repository_root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"cannot record unavailable file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExportError(f"cannot stat file: {path}") from exc
    try:
        display_path = _repo_relative(path, repository_root=repository_root)
    except ExportError:
        display_path = str(path.resolve())
    return {
        "path": display_path,
        "bytes": int(size),
        "sha256": _sha256_file(path),
    }


def _current_commit(repository_root: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExportError("unable to resolve exporter execution commit") from exc
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ExportError("exporter execution commit is not a full hash")
    return commit


def _resolve_bound_file(value: Any, *, repository_root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{description} path is absent")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise ExportError(f"{description} is unavailable: {candidate}")
    return candidate.resolve()


def _expected_artifact_path(receipt_path: Path) -> Path:
    suffix = ".run.json"
    if not receipt_path.name.endswith(suffix):
        raise ExportError(f"receipt has unexpected filename: {receipt_path}")
    return receipt_path.with_name(receipt_path.name[: -len(suffix)] + ".safetensors")


def _assert_same_except(
    before: Any,
    after: Any,
    *,
    allowed_paths: frozenset[str],
    path: tuple[str, ...] = (),
) -> None:
    path_key = ".".join(path)
    if path_key in allowed_paths:
        return
    if type(before) is not type(after):
        raise ExportError(f"unapproved type change at {path_key or '<root>'}")
    if isinstance(before, Mapping):
        if set(before) != set(after):
            raise ExportError(f"unapproved key change at {path_key or '<root>'}")
        for key in before:
            _assert_same_except(
                before[key],
                after[key],
                allowed_paths=allowed_paths,
                path=path + (str(key),),
            )
        return
    if isinstance(before, list):
        if len(before) != len(after):
            raise ExportError(f"unapproved list length change at {path_key or '<root>'}")
        for index, (left, right) in enumerate(zip(before, after)):
            _assert_same_except(
                left,
                right,
                allowed_paths=allowed_paths,
                path=path + (str(index),),
            )
        return
    if before != after:
        raise ExportError(f"unapproved value change at {path_key or '<root>'}")


def _expected_keys() -> set[tuple[str, str]]:
    return {
        (cell_id, method_id)
        for cell_id in EXPECTED_CELL_IDS
        for method_id in METHOD_IDS
    }


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    schema: str,
    map_key: str,
    description: str,
) -> dict[str, Any]:
    if manifest.get("schema") != schema:
        raise ExportError(f"{description} manifest schema changed")
    if manifest.get("task_id") != TASK_ID:
        raise ExportError(f"{description} manifest task ID changed")
    if manifest.get("truth_opened") is not False:
        raise ExportError(f"{description} manifest opened truth")
    if tuple(manifest.get("method_ids", ())) != METHOD_IDS:
        raise ExportError(f"{description} manifest method order changed")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or set(cells) != set(EXPECTED_CELL_IDS):
        raise ExportError(f"{description} manifest cell set changed")
    entries = manifest.get(map_key)
    if not isinstance(entries, Mapping):
        raise ExportError(f"{description} manifest has no {map_key} map")
    expected_text_keys = {f"{cell}::{method}" for cell, method in _expected_keys()}
    if set(entries) != expected_text_keys:
        raise ExportError(f"{description} manifest entry set changed")
    return dict(manifest)


def _discover_source_receipts(
    source_root: Path,
    *,
    repository_root: Path,
) -> dict[tuple[str, str], tuple[Path, dict[str, Any]]]:
    paths = sorted(source_root.rglob("*.run.json"))
    if len(paths) != len(_expected_keys()):
        raise ExportError(
            f"expected {len(_expected_keys())} source receipts, found {len(paths)}"
        )
    discovered: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        value = _load_json(path, description="source prediction receipt")
        cell_id = value.get("cell_id")
        method_id = value.get("method_id")
        key = (cell_id, method_id)
        if key not in _expected_keys():
            raise ExportError(f"source receipt has unknown cell/method: {path}")
        if key in discovered:
            raise ExportError(f"duplicate source receipt: {cell_id}/{method_id}")
        style, condition = cell_id.split("__", 1)
        expected = source_root / style / condition / f"{method_id}.run.json"
        if path.resolve() != expected.resolve():
            raise ExportError(f"source receipt path does not match its binding: {path}")
        # Resolve the artifact binding now, while the receipt still names the
        # immutable v1 path.  The actual hash/size check is done below.
        _resolve_bound_file(
            (value.get("prediction_artifact") or {}).get("path")
            if isinstance(value.get("prediction_artifact"), Mapping)
            else None,
            repository_root=repository_root,
            description=f"{cell_id}/{method_id} prediction artifact",
        )
        discovered[key] = (path, value)
    if set(discovered) != _expected_keys():
        raise ExportError("source receipt matrix is incomplete")
    return discovered


def _validate_source_run_evidence(
    evidence: Mapping[str, Any], *, execution_commit: str
) -> None:
    if evidence.get("schema") != RUN_EVIDENCE_SCHEMA:
        raise ExportError("source run evidence schema changed")
    if evidence.get("task_id") != TASK_ID:
        raise ExportError("source run evidence task ID changed")
    if evidence.get("status") != "PUBLIC_PREDICTION_MATRIX_COMPLETE_NO_TRUTH":
        raise ExportError("source run evidence is not a completed public run")
    if evidence.get("git_commit") != execution_commit:
        raise ExportError("source run evidence commit differs from exporter HEAD")
    for key in (
        "truth_opened",
        "source_text_loaded",
        "target_labels_loaded",
        "future_activation_reads",
    ):
        if evidence.get(key) is not False:
            raise ExportError(f"source run evidence has unsafe {key} marker")
    if evidence.get("prediction_count") != len(_expected_keys()):
        raise ExportError("source run evidence prediction count changed")
    if evidence.get("timing_count") != len(_expected_keys()):
        raise ExportError("source run evidence timing count changed")


def _validate_source_artifact(
    descriptor: Mapping[str, Any],
    *,
    receipt_path: Path,
    repository_root: Path,
) -> tuple[Path, dict[str, Any]]:
    binding = descriptor.get("prediction_artifact")
    if not isinstance(binding, Mapping):
        raise ExportError(f"prediction artifact binding is absent: {receipt_path}")
    expected_path = _expected_artifact_path(receipt_path).resolve()
    bound_path = _resolve_bound_file(
        binding.get("path"),
        repository_root=repository_root,
        description=f"{descriptor.get('cell_id')}/{descriptor.get('method_id')} prediction artifact",
    )
    if bound_path != expected_path:
        raise ExportError(f"prediction artifact path changed: {receipt_path}")
    relative_path = _resolve_bound_file(
        descriptor.get("artifact_relative_to_root"),
        repository_root=repository_root,
        description=f"{descriptor.get('cell_id')}/{descriptor.get('method_id')} relative artifact",
    )
    if relative_path != expected_path:
        raise ExportError(f"relative artifact path changed: {receipt_path}")
    actual = _file_record(expected_path, repository_root=repository_root)
    if binding.get("bytes") != actual["bytes"] or binding.get("sha256") != actual["sha256"]:
        raise ExportError(f"prediction artifact hash changed: {receipt_path}")
    tensor_digest = descriptor.get("prediction_sha256")
    if not isinstance(tensor_digest, str) or re.fullmatch(r"[0-9a-f]{64}", tensor_digest) is None:
        raise ExportError(f"prediction tensor digest is malformed: {receipt_path}")
    return expected_path, actual


def _correct_descriptor(
    descriptor: Mapping[str, Any],
    *,
    receipt_path: Path,
    destination_receipt_path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_id = descriptor.get("cell_id")
    method_id = descriptor.get("method_id")
    if not isinstance(cell_id, str) or not isinstance(method_id, str):
        raise ExportError(f"source receipt lacks cell/method binding: {receipt_path}")
    if descriptor.get("schema") != LEGACY_RECEIPT_SCHEMA:
        raise ExportError(f"source receipt schema is not the known legacy value: {receipt_path}")
    artifact_path, source_artifact = _validate_source_artifact(
        descriptor,
        receipt_path=receipt_path,
        repository_root=repository_root,
    )
    destination_artifact = _expected_artifact_path(destination_receipt_path).resolve()
    destination_relative = _repo_relative(
        destination_artifact, repository_root=repository_root
    )
    corrected = copy.deepcopy(dict(descriptor))
    corrected["schema"] = PREDICTION_SCHEMA
    corrected_artifact = dict(corrected["prediction_artifact"])
    corrected_artifact["path"] = destination_relative
    corrected["prediction_artifact"] = corrected_artifact
    corrected["artifact_relative_to_root"] = destination_relative
    _assert_same_except(
        descriptor,
        corrected,
        allowed_paths=ALLOWED_DESCRIPTOR_CHANGES,
    )
    try:
        validate_prediction_descriptor(
            corrected,
            cell_id=cell_id,
            method_id=method_id,
        )
    except Exception as exc:
        raise ExportError(
            f"corrected descriptor fails the TRR-0005 contract: {receipt_path}"
        ) from exc
    return corrected, {
        "cell_id": cell_id,
        "method_id": method_id,
        "source_receipt": _file_record(receipt_path, repository_root=repository_root),
        "source_artifact": source_artifact,
        "source_artifact_path": str(artifact_path),
        "destination_receipt_path": str(destination_receipt_path),
        "destination_artifact_path": str(destination_artifact),
        "destination_relative_artifact": destination_relative,
    }


def _copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise ExportError(f"refusing to overwrite copied artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
            destination_handle.flush()
    except OSError as exc:
        raise ExportError(f"unable to copy prediction artifact: {source}") from exc
    source_digest = _sha256_file(source)
    destination_digest = _sha256_file(destination)
    if source_digest != destination_digest or source.stat().st_size != destination.stat().st_size:
        raise ExportError(f"prediction artifact copy is not byte-identical: {source}")
    return {
        "bytes": int(destination.stat().st_size),
        "sha256": destination_digest,
    }


def _manifest_copy(
    manifest: Mapping[str, Any],
    *,
    map_key: str,
    source_entries: Mapping[str, Mapping[str, Any]],
    corrected_entries: Mapping[str, Mapping[str, Any]],
    description: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(manifest))
    original_entries = manifest[map_key]
    for key, source_descriptor in source_entries.items():
        if original_entries.get(key) != source_descriptor:
            raise ExportError(f"{description} entry differs from its raw receipt: {key}")
    result[map_key] = {
        key: copy.deepcopy(corrected_entries[key]) for key in original_entries
    }
    for key, value in manifest.items():
        if key != map_key and result.get(key) != value:
            raise ExportError(f"{description} changed outside its entry map: {key}")
    return result


def _prepare_export(
    *,
    source_root: Path,
    output_root: Path,
    repository_root: Path,
    execution_commit: str,
    exporter_path: Path,
) -> dict[str, Any]:
    source_receipts = _discover_source_receipts(
        source_root,
        repository_root=repository_root,
    )
    source_prediction_manifest = _validate_manifest(
        _load_json(source_root / "predictions.json", description="source prediction"),
        schema=PREDICTION_MANIFEST_SCHEMA,
        map_key=PREDICTION_MAP_KEY,
        description="source prediction",
    )
    source_timing_manifest = _validate_manifest(
        _load_json(source_root / "timings.json", description="source timing"),
        schema=TIMING_MANIFEST_SCHEMA,
        map_key=TIMING_MAP_KEY,
        description="source timing",
    )
    run_evidence_path = source_root / "run_evidence.json"
    run_evidence = _load_json(run_evidence_path, description="source run evidence")
    _validate_source_run_evidence(run_evidence, execution_commit=execution_commit)
    source_artifact_paths: set[Path] = set()
    expected_entries = _expected_keys()
    corrected_entries: dict[str, dict[str, Any]] = {}
    source_entries: dict[str, dict[str, Any]] = {}
    item_info: list[dict[str, Any]] = []
    for key in sorted(expected_entries):
        receipt_path, source_descriptor = source_receipts[key]
        cell_id, method_id = key
        text_key = f"{cell_id}::{method_id}"
        source_entries[text_key] = source_descriptor
        destination_receipt_path = output_root / receipt_path.relative_to(source_root)
        corrected, info = _correct_descriptor(
            source_descriptor,
            receipt_path=receipt_path,
            destination_receipt_path=destination_receipt_path,
            repository_root=repository_root,
        )
        corrected_entries[text_key] = corrected
        source_artifact_paths.add(Path(info["source_artifact_path"]).resolve())
        item_info.append(info)
    discovered_artifacts = {
        path.resolve() for path in source_root.rglob("*.safetensors") if path.is_file()
    }
    if discovered_artifacts != source_artifact_paths:
        raise ExportError("source artifact set does not match the 32 receipt bindings")
    source_prediction_manifest_copy = _manifest_copy(
        source_prediction_manifest,
        map_key=PREDICTION_MAP_KEY,
        source_entries=source_entries,
        corrected_entries=corrected_entries,
        description="source prediction",
    )
    source_timing_manifest_copy = _manifest_copy(
        source_timing_manifest,
        map_key=TIMING_MAP_KEY,
        source_entries=source_entries,
        corrected_entries=corrected_entries,
        description="source timing",
    )
    # The copies above are intentionally built before creating the destination,
    # so malformed public metadata cannot leave a partial export directory.
    return {
        "source_receipts": source_receipts,
        "source_prediction_manifest": source_prediction_manifest,
        "source_timing_manifest": source_timing_manifest,
        "source_prediction_manifest_copy": source_prediction_manifest_copy,
        "source_timing_manifest_copy": source_timing_manifest_copy,
        "run_evidence": run_evidence,
        "run_evidence_path": run_evidence_path,
        "item_info": item_info,
        "source_artifact_paths": source_artifact_paths,
        "exporter_path": exporter_path,
        "execution_commit": execution_commit,
    }


def _relative_root(path: Path, *, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ExportError(f"export root is outside repository root: {path}") from exc


def _provenance(
    prepared: Mapping[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    repository_root: Path,
    destination_receipts: list[dict[str, Any]],
    destination_artifacts: list[dict[str, Any]],
    destination_prediction_manifest: Path,
    destination_timing_manifest: Path,
) -> dict[str, Any]:
    source_receipts = prepared["source_receipts"]
    source_receipt_records = []
    for key in sorted(source_receipts):
        source_receipt_records.append(
            {
                "cell_id": key[0],
                "method_id": key[1],
                "file": _file_record(
                    source_receipts[key][0], repository_root=repository_root
                ),
            }
        )
    source_artifact_records = []
    for key in sorted(source_receipts):
        info = next(
            item
            for item in prepared["item_info"]
            if (item["cell_id"], item["method_id"]) == key
        )
        source_artifact_records.append(
            {
                "cell_id": key[0],
                "method_id": key[1],
                "file": info["source_artifact"],
            }
        )
    exporter_path = Path(prepared["exporter_path"]).resolve()
    return {
        "schema": EXPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "METADATA_ONLY_CONTRACT_EXPORT_NO_TRUTH",
        "execution_commit": prepared["execution_commit"],
        "exporter": _file_record(exporter_path, repository_root=repository_root),
        "source": {
            "root": _relative_root(source_root, repository_root=repository_root),
            "predictions_manifest": _file_record(
                source_root / "predictions.json", repository_root=repository_root
            ),
            "timings_manifest": _file_record(
                source_root / "timings.json", repository_root=repository_root
            ),
            "run_evidence": _file_record(
                prepared["run_evidence_path"], repository_root=repository_root
            ),
            "run_evidence_rewritten": False,
            "prediction_receipts": source_receipt_records,
            "prediction_artifacts": source_artifact_records,
        },
        "destination": {
            "root": _relative_root(output_root, repository_root=repository_root),
            "predictions_manifest": _file_record(
                destination_prediction_manifest, repository_root=repository_root
            ),
            "timings_manifest": _file_record(
                destination_timing_manifest, repository_root=repository_root
            ),
            "prediction_receipts": destination_receipts,
            "prediction_artifacts": destination_artifacts,
        },
        "allowed_descriptor_changes": sorted(ALLOWED_DESCRIPTOR_CHANGES),
        "descriptor_change_policy": {
            "schema_from": LEGACY_RECEIPT_SCHEMA,
            "schema_to": PREDICTION_SCHEMA,
            "artifact_paths_rebased_to_destination_root": True,
            "all_other_descriptor_fields_semantically_equal": True,
            "all_timing_values_preserved": True,
            "prediction_tensor_digests_preserved": True,
        },
        "binary_copy_policy": {
            "copy_mode": "byte_for_byte",
            "prediction_artifact_count": len(destination_artifacts),
            "destination_bytes_and_sha256_verified": True,
            "tensor_contents_loaded": False,
        },
        "preserved_execution_record": {
            "path": _repo_relative(
                prepared["run_evidence_path"], repository_root=repository_root
            ),
            "status": prepared["run_evidence"]["status"],
            "raw_source_left_untouched": True,
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "future_activation_reads": False,
    }


def export_prediction_contract(
    source_root: Path,
    output_root: Path,
    *,
    repository_root: Path,
    execution_commit: str | None = None,
    exporter_path: Path | None = None,
) -> dict[str, Any]:
    """Copy v1 prediction bytes and export corrected v2 metadata.

    All source validation and descriptor comparisons happen before the new
    output directory is created.  The destination is create-only; a partial
    output on a later I/O failure receives a preserved ``failure.json``.
    """

    repository_root = repository_root.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root:
        raise ExportError("source and destination roots must differ")
    if source_root.is_symlink() or not source_root.is_dir():
        raise ExportError(f"source prediction root is unavailable: {source_root}")
    if output_root.exists() or output_root.is_symlink():
        raise ExportError(f"export destination must be new: {output_root}")
    _relative_root(source_root, repository_root=repository_root)
    _relative_root(output_root, repository_root=repository_root)
    commit = execution_commit or _current_commit(repository_root)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ExportError("export execution commit is not a full hash")
    script_path = (exporter_path or Path(__file__)).expanduser().resolve()
    if script_path.is_symlink() or not script_path.is_file():
        raise ExportError(f"exporter source is unavailable: {script_path}")
    prepared = _prepare_export(
        source_root=source_root,
        output_root=output_root,
        repository_root=repository_root,
        execution_commit=commit,
        exporter_path=script_path,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        destination_receipt_records: list[dict[str, Any]] = []
        destination_artifact_records: list[dict[str, Any]] = []
        corrected_entries = {
            f"{item['cell_id']}::{item['method_id']}": None
            for item in prepared["item_info"]
        }
        source_entries = prepared["source_prediction_manifest"][PREDICTION_MAP_KEY]
        corrected_by_key = prepared["source_prediction_manifest_copy"][PREDICTION_MAP_KEY]
        for item in sorted(
            prepared["item_info"], key=lambda value: (value["cell_id"], value["method_id"])
        ):
            key = f"{item['cell_id']}::{item['method_id']}"
            receipt_destination = Path(item["destination_receipt_path"])
            artifact_destination = Path(item["destination_artifact_path"])
            source_artifact = Path(item["source_artifact_path"])
            copied = _copy_exact(source_artifact, artifact_destination)
            source_record = item["source_artifact"]
            if copied["bytes"] != source_record["bytes"] or copied["sha256"] != source_record["sha256"]:
                raise ExportError(f"copied artifact record differs: {key}")
            destination_artifact_records.append(
                {
                    "cell_id": item["cell_id"],
                    "method_id": item["method_id"],
                    "file": _file_record(
                        artifact_destination, repository_root=repository_root
                    ),
                }
            )
            _json_create(receipt_destination, corrected_by_key[key])
            destination_receipt_records.append(
                {
                    "cell_id": item["cell_id"],
                    "method_id": item["method_id"],
                    "file": _file_record(
                        receipt_destination, repository_root=repository_root
                    ),
                }
            )
        destination_prediction_manifest = output_root / "predictions.json"
        destination_timing_manifest = output_root / "timings.json"
        _json_create(
            destination_prediction_manifest,
            prepared["source_prediction_manifest_copy"],
        )
        _json_create(
            destination_timing_manifest,
            prepared["source_timing_manifest_copy"],
        )
        provenance = _provenance(
            prepared,
            source_root=source_root,
            output_root=output_root,
            repository_root=repository_root,
            destination_receipts=destination_receipt_records,
            destination_artifacts=destination_artifact_records,
            destination_prediction_manifest=destination_prediction_manifest,
            destination_timing_manifest=destination_timing_manifest,
        )
        provenance_path = output_root / "export_provenance.json"
        _json_create(provenance_path, provenance)
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr0005-prediction-contract-export-failure.v1",
            "task_id": TASK_ID,
            "status": "FAILED_PRESERVED_NO_TRUTH",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "execution_commit": commit,
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "future_activation_reads": False,
        }
        try:
            _json_create(output_root / "failure.json", failure)
        except Exception:
            pass
        if isinstance(exc, ExportError):
            raise
        raise ExportError("prediction contract export failed") from exc
    return {
        "schema": EXPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "METADATA_ONLY_CONTRACT_EXPORT_NO_TRUTH",
        "execution_commit": commit,
        "source_root": _relative_root(source_root, repository_root=repository_root),
        "output_root": _relative_root(output_root, repository_root=repository_root),
        "prediction_receipts": len(prepared["item_info"]),
        "prediction_artifacts": len(prepared["item_info"]),
        "predictions_manifest": _repo_relative(
            output_root / "predictions.json", repository_root=repository_root
        ),
        "timings_manifest": _repo_relative(
            output_root / "timings.json", repository_root=repository_root
        ),
        "provenance": _repo_relative(
            output_root / "export_provenance.json", repository_root=repository_root
        ),
        "truth_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.expanduser().resolve()
    try:
        result = export_prediction_contract(
            args.source_root,
            args.output_root,
            repository_root=root,
        )
    except (ExportError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-0005 prediction contract export failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ALLOWED_DESCRIPTOR_CHANGES",
    "EXPORT_SCHEMA",
    "ExportError",
    "LEGACY_RECEIPT_SCHEMA",
    "PREDICTION_MANIFEST_SCHEMA",
    "TIMING_MANIFEST_SCHEMA",
    "export_prediction_contract",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
