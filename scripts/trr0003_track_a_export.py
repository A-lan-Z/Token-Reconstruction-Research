#!/usr/bin/env python3
"""Export immutable Track A iteration diagnostics into the common pilot layout.

Track A deliberately records one checkpoint-only method and a fixed iteration
ladder.  The common TRR-0003 scorer needs one method ID per selected iteration,
so this utility performs a serialization-only alias export.  It reads only the
public diagnostic metadata/tensors and never opens a truth or source-label
asset.  The raw diagnostics remain untouched; each alias records the source
artifact, source evidence, tensor digests, and exporter bytes in its binding.

The exporter is intentionally strict: it discovers exactly one completed
Track A evidence file for each of the four shared panel cells, verifies every
ladder artifact before writing any alias, and refuses to overwrite output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from token_reconstruction.footing import (
    CUT_DEPTH,
    HIDDEN_SIZE,
    INVALID_TOKEN_ID,
    PANEL_SCHEMA,
    PREDICTION_SCHEMA,
    TASK_ID,
    FootingError,
    expected_cell_ids,
    expected_prediction_path,
    external_file_record,
    file_record,
    load_all_cells,
    load_panel,
    sha256_file,
    tensor_sha256,
    validate_prediction_artifact,
)


BASE_METHOD_ID = "checkpoint_reverse_fixed_point_euclidean_k16"
EXPORT_SCHEMA = "token-reconstruction.trr0003-track-a-export.v1"
MAP_SCHEMA = "token-reconstruction.trr0003-track-a-export-manifest.v1"
EVIDENCE_SCHEMA = "token-reconstruction.trr0003-track-a-evidence.v1"
DEFAULT_DIAGNOSTIC_ROOT = Path("outputs/TRR-0003/track_a_diagnostics")
DEFAULT_OUTPUT_ROOT = Path("outputs/TRR-0003/track_a_export_v1")
DEFAULT_ITERATIONS = (0, 1, 2, 4, 8, 16, 32)
BOS_TOKEN_ID = 128000
VOCAB_SIZE = 128256


class ExportError(RuntimeError):
    """Raised when an immutable Track A export fails closed."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_load(path: Path, *, description: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"{description} must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"{description} is invalid JSON: {path}") from exc


def _json_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ExportError(f"refusing to overwrite export manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()


def _current_commit(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExportError("unable to resolve exporter execution commit") from exc
    if len(value) != 40:
        raise ExportError("exporter execution commit is not a full hash")
    return value


def _repo_path(value: Any, *, root: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{description} path is absent")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"{description} is unavailable: {path}")
    return path.resolve()


def _record_for_any_path(path: Path, *, root: Path) -> dict[str, Any]:
    """Record a source asset as repo-relative when possible, external otherwise."""

    try:
        return file_record(path, repository_root=root)
    except FootingError:
        return external_file_record(path)


def _normalize_binding_asset(
    value: Any, *, root: Path, description: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportError(f"{description} binding is malformed")
    path = _repo_path(value.get("path"), root=root, description=description)
    normalized = _record_for_any_path(path, root=root)
    # A source binding is an input commitment.  Refuse to silently change it
    # while converting absolute in-repo paths into the common relative form.
    for field in ("bytes", "sha256"):
        if value.get(field) != normalized[field]:
            raise ExportError(f"{description} changed while normalizing")
    return normalized


def _normalize_source_binding(
    source_binding: Mapping[str, Any], *, root: Path, adapter_path: Path, commit: str
) -> dict[str, Any]:
    if not isinstance(source_binding, Mapping):
        raise ExportError("Track A source binding is absent")
    result = dict(source_binding)
    panel = source_binding.get("panel")
    if not isinstance(panel, Mapping):
        raise ExportError("Track A source binding omits panel")
    state = source_binding.get("method_state")
    code = source_binding.get("code")
    if not isinstance(state, list) or not state:
        raise ExportError("Track A source binding omits method state")
    if not isinstance(code, list) or not code:
        raise ExportError("Track A source binding omits code")
    result["panel"] = _normalize_binding_asset(panel, root=root, description="Track A panel")
    result["method_state"] = [
        _normalize_binding_asset(item, root=root, description="Track A method state")
        for item in state
    ]
    result["code"] = [
        _normalize_binding_asset(item, root=root, description="Track A source code")
        for item in code
    ]
    # The adapter is executable serialization code.  Include its exact bytes
    # in the top-level binding; the original source binding stays nested below.
    result["code"].append(file_record(adapter_path, repository_root=root))
    result["code_commit"] = commit
    return result


def _geometry(cell: Any) -> dict[str, int]:
    return {
        "records": int(cell.records),
        "sequence_tokens": int(cell.sequence_tokens),
        "hidden_size": HIDDEN_SIZE,
        "cut_depth": CUT_DEPTH,
    }


def _safe_cell_evidence(
    path: Path,
    *,
    cell: Any,
    panel_sha: str,
    iterations: Sequence[int],
) -> dict[str, Any]:
    evidence = _json_load(path, description="Track A evidence")
    if not isinstance(evidence, Mapping):
        raise ExportError(f"Track A evidence root is malformed: {path}")
    if evidence.get("schema") != EVIDENCE_SCHEMA or evidence.get("task_id") != TASK_ID:
        raise ExportError(f"Track A evidence identity changed: {path}")
    if evidence.get("method_id") != BASE_METHOD_ID:
        raise ExportError(f"Track A evidence method changed: {path}")
    if evidence.get("truth_opened") is not False:
        raise ExportError(f"Track A evidence opened truth: {path}")
    if evidence.get("source_material_included") is not False:
        raise ExportError(f"Track A evidence includes source material: {path}")
    if evidence.get("status") != "PREDICTIONS_FROZEN":
        raise ExportError(f"Track A evidence is not frozen: {path}")
    if evidence.get("canonical_comparison_complete") is not False:
        raise ExportError(f"Track A evidence mislabels canonical completeness: {path}")
    panel = evidence.get("panel")
    if not isinstance(panel, Mapping):
        raise ExportError(f"Track A evidence panel record is absent: {path}")
    expected_panel = {
        "cell_id": cell.cell_id,
        "condition": cell.condition,
        "cut_depth": CUT_DEPTH,
        "hidden_size": HIDDEN_SIZE,
        "records": cell.records,
        "sequence_tokens": cell.sequence_tokens,
        "style": cell.style,
        "selected_record_indices": list(range(cell.records)),
        "sha256": panel_sha,
    }
    for key, expected in expected_panel.items():
        if panel.get(key) != expected:
            raise ExportError(f"Track A evidence panel field changed ({key}): {path}")
    expected_ids = list(cell.record_ids)
    if panel.get("record_ids") != expected_ids:
        raise ExportError(f"Track A evidence record order changed: {path}")
    if panel.get("observation_dtype") != "torch.bfloat16":
        raise ExportError(f"Track A evidence observation dtype changed: {path}")
    if not isinstance(panel.get("target_weight_available_to_method"), bool):
        raise ExportError(f"Track A target-weight availability marker is malformed: {path}")
    if cell.condition == "public_lora_2601" and panel.get("target_weight_available_to_method") is not False:
        raise ExportError(f"Track A shifted evidence exposes target weights: {path}")
    if evidence.get("final_iteration") != max(iterations):
        raise ExportError(f"Track A final iteration changed: {path}")
    ladder = evidence.get("iterations")
    if not isinstance(ladder, list) or len(ladder) != len(iterations):
        raise ExportError(f"Track A iteration ladder is incomplete: {path}")
    seen: list[int] = []
    for entry in ladder:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("iterations"), int):
            raise ExportError(f"Track A iteration evidence is malformed: {path}")
        seen.append(int(entry["iterations"]))
    if seen != list(iterations):
        raise ExportError(f"Track A iteration ladder changed: {path}")
    if not isinstance(evidence.get("method_state_binding"), Mapping):
        raise ExportError(f"Track A method binding is absent: {path}")
    return dict(evidence)


def _read_source(
    path: Path,
    *,
    cell: Any,
    panel_sha: str,
    iteration: int,
    evidence: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"Track A source artifact is unavailable: {path}")
    ladder = evidence["iterations"]
    entry = next((row for row in ladder if row.get("iterations") == iteration), None)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("artifact"), Mapping):
        raise ExportError(f"Track A source iteration is absent: {path}")
    source_record = entry["artifact"]
    if source_record.get("path") != str(path.resolve()):
        raise ExportError(f"Track A artifact path binding changed: {path}")
    if source_record.get("bytes") != path.stat().st_size or source_record.get("sha256") != sha256_file(path):
        raise ExportError(f"Track A artifact hash changed: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = set(handle.keys())
            expected_keys = {
                "predictions",
                "candidates",
                "candidate_scores",
                "continuous_residual",
                "discrete_residual",
            }
            if keys != expected_keys:
                raise ExportError(f"Track A tensor fields changed: {path}")
            if metadata.get("schema") != EVIDENCE_SCHEMA or metadata.get("task_id") != TASK_ID:
                raise ExportError(f"Track A source artifact identity changed: {path}")
            if metadata.get("method_id") != BASE_METHOD_ID:
                raise ExportError(f"Track A source method changed: {path}")
            if metadata.get("panel_sha256") != panel_sha:
                raise ExportError(f"Track A source panel changed: {path}")
            if metadata.get("cell_id") != cell.cell_id or metadata.get("style") != cell.style or metadata.get("condition") != cell.condition:
                raise ExportError(f"Track A source cell changed: {path}")
            if metadata.get("iteration") != str(iteration):
                raise ExportError(f"Track A source iteration changed: {path}")
            if metadata.get("truth_opened") != "false":
                raise ExportError(f"Track A source opened truth: {path}")
            geometry = json.loads(str(metadata.get("geometry_json", "{}")))
            if geometry != _geometry(cell):
                raise ExportError(f"Track A source geometry changed: {path}")
            binding_json = metadata.get("binding_json")
            if not isinstance(binding_json, str):
                raise ExportError(f"Track A source binding is absent: {path}")
            source_binding = json.loads(binding_json)
            tensors = {key: handle.get_tensor(key).contiguous() for key in expected_keys}
            metadata_copy = {str(key): str(value) for key, value in metadata.items()}
    except ExportError:
        raise
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise ExportError(f"Track A source artifact is unreadable: {path}") from exc
    mask = cell.attention_mask.to(torch.bool)
    predictions = tensors["predictions"]
    candidates = tensors["candidates"]
    scores = tensors["candidate_scores"]
    for name, value in (("predictions", predictions), ("candidates", candidates)):
        if name == "predictions":
            if tuple(value.shape) != tuple(cell.attention_mask.shape) or value.ndim != 2:
                raise ExportError(f"Track A {name} geometry changed: {path}")
        elif value.ndim != 3 or tuple(value.shape[:2]) != tuple(cell.attention_mask.shape) or value.shape[2] != 16:
            raise ExportError(f"Track A {name} geometry changed: {path}")
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ExportError(f"Track A {name} dtype changed: {path}")
        if name == "predictions" and value[:, 0].ne(BOS_TOKEN_ID).any().item():
            raise ExportError(f"Track A {name} BOS changed: {path}")
        if value[mask].lt(0).any().item() or value[mask].ge(VOCAB_SIZE).any().item():
            raise ExportError(f"Track A {name} active token range changed: {path}")
        if value[~mask].ne(INVALID_TOKEN_ID).any().item():
            raise ExportError(f"Track A {name} padding marker changed: {path}")
    if tuple(scores.shape) != tuple(candidates.shape) or scores.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ExportError(f"Track A candidate score geometry or dtype changed: {path}")
    if not torch.isfinite(scores[mask]).all().item():
        raise ExportError(f"Track A candidate score is non-finite: {path}")
    for name in ("continuous_residual", "discrete_residual"):
        value = tensors[name]
        if tuple(value.shape) != tuple(cell.attention_mask.shape) or not value.dtype.is_floating_point:
            raise ExportError(f"Track A residual geometry or dtype changed: {path}")
        if not torch.isfinite(value[mask]).all().item():
            raise ExportError(f"Track A residual contains non-finite values: {path}")
    source_evidence_binding = evidence.get("method_state_binding")
    if json.dumps(source_binding, sort_keys=True) != json.dumps(source_evidence_binding, sort_keys=True):
        raise ExportError(f"Track A source binding differs from evidence: {path}")
    selected = {name: tensors[name] for name in ("predictions", "candidates", "candidate_scores")}
    digests = {name: tensor_sha256(value) for name, value in selected.items()}
    return selected, metadata_copy, {"binding": source_binding, "tensor_sha256": digests}


def _alias_binding(
    *,
    source_binding: Mapping[str, Any],
    root: Path,
    adapter_path: Path,
    commit: str,
    source_artifact: Path,
    source_evidence: Path,
    source_iteration: int,
    source_tensor_sha256: Mapping[str, str],
) -> dict[str, Any]:
    normalized = _normalize_source_binding(
        source_binding, root=root, adapter_path=adapter_path, commit=commit
    )
    normalized["track_a_export"] = {
        "schema": EXPORT_SCHEMA,
        "serialization_only": True,
        "source_method_id": BASE_METHOD_ID,
        "source_iteration": source_iteration,
        "source_artifact": _record_for_any_path(source_artifact, root=root),
        "source_artifact_sha256": sha256_file(source_artifact),
        "source_evidence": _record_for_any_path(source_evidence, root=root),
        "source_evidence_sha256": sha256_file(source_evidence),
        "source_tensor_sha256": dict(source_tensor_sha256),
        "exporter": file_record(adapter_path, repository_root=root),
        "exporter_execution_commit": commit,
    }
    return normalized


def _source_artifact_path(evidence: Mapping[str, Any], *, iteration: int, root: Path) -> Path:
    for entry in evidence["iterations"]:
        if isinstance(entry, Mapping) and entry.get("iterations") == iteration:
            artifact = entry.get("artifact")
            if isinstance(artifact, Mapping):
                return _repo_path(artifact.get("path"), root=root, description="Track A source artifact")
    raise ExportError(f"Track A iteration artifact is absent: {iteration}")


def _export(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    panel_path = (root / args.panel if not args.panel.is_absolute() else args.panel).resolve()
    diagnostic_root = (root / args.diagnostic_root if not args.diagnostic_root.is_absolute() else args.diagnostic_root).resolve()
    output_root = (root / args.output_root if not args.output_root.is_absolute() else args.output_root).resolve()
    adapter_path = Path(__file__).resolve()
    if adapter_path.is_symlink() or not adapter_path.is_file():
        raise ExportError("exporter source is unavailable")
    if output_root.exists() or output_root.is_symlink():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise ExportError(f"export output must be a new empty directory: {output_root}")
    else:
        output_root.mkdir(parents=True)
    panel = load_panel(panel_path, repository_root=root)
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise ExportError("panel identity changed")
    cells = load_all_cells(panel, repository_root=root)
    cells_by_id = {cell.cell_id: cell for cell in cells}
    panel_sha = sha256_file(panel_path)
    iterations = tuple(int(value) for value in args.iterations)
    if not iterations or tuple(sorted(set(iterations))) != iterations:
        raise ExportError("iterations must be a strictly increasing non-empty ladder")
    if any(value < 0 for value in iterations):
        raise ExportError("iterations must be non-negative")
    evidence_paths = sorted(diagnostic_root.rglob("evidence.json"))
    if len(evidence_paths) != len(expected_cell_ids()):
        raise ExportError(f"expected four Track A evidence files, found {len(evidence_paths)}")
    source_by_cell: dict[str, tuple[Path, dict[str, Any]]] = {}
    source_artifacts: dict[str, dict[int, tuple[Path, dict[str, torch.Tensor], dict[str, str], dict[str, Any]]]] = {}
    for evidence_path in evidence_paths:
        # Discover the cell only from the signed evidence panel record; never
        # trust the directory names because the diagnostic root is user input.
        discovered = _json_load(evidence_path, description="Track A evidence")
        if not isinstance(discovered, Mapping) or not isinstance(discovered.get("panel"), Mapping):
            raise ExportError(f"Track A evidence panel record is absent: {evidence_path}")
        cell_id = discovered["panel"].get("cell_id")
        if cell_id not in cells_by_id or cell_id in source_by_cell:
            raise ExportError(f"Track A evidence cell set is invalid: {evidence_path}")
        cell = cells_by_id[cell_id]
        evidence = _safe_cell_evidence(
            evidence_path, cell=cell, panel_sha=panel_sha, iterations=iterations
        )
        source_by_cell[cell_id] = (evidence_path, evidence)
        source_artifacts[cell_id] = {}
        source_binding: Mapping[str, Any] | None = None
        for iteration in iterations:
            artifact_path = _source_artifact_path(evidence, iteration=iteration, root=root)
            selected, metadata, extra = _read_source(
                artifact_path,
                cell=cell,
                panel_sha=panel_sha,
                iteration=iteration,
                evidence=evidence,
            )
            binding = extra["binding"]
            if source_binding is None:
                source_binding = binding
            elif json.dumps(source_binding, sort_keys=True) != json.dumps(binding, sort_keys=True):
                raise ExportError(f"Track A binding changed across iterations: {cell_id}")
            source_artifacts[cell_id][iteration] = (artifact_path, selected, metadata, extra)
    if set(source_by_cell) != set(expected_cell_ids()):
        raise ExportError("Track A evidence does not cover all panel cells")
    commit = _current_commit(root)
    manifest_methods: dict[str, Any] = {}
    aliases = [f"{BASE_METHOD_ID}_i{iteration:03d}" for iteration in iterations]
    for alias, iteration in zip(aliases, iterations):
        method_cells: dict[str, Any] = {}
        method_bindings: dict[str, dict[str, Any]] = {}
        for cell in cells:
            evidence_path, evidence = source_by_cell[cell.cell_id]
            artifact_path, tensors, metadata, extra = source_artifacts[cell.cell_id][iteration]
            binding = _alias_binding(
                source_binding=extra["binding"],
                root=root,
                adapter_path=adapter_path,
                commit=commit,
                source_artifact=artifact_path,
                source_evidence=evidence_path,
                source_iteration=iteration,
                source_tensor_sha256=extra["tensor_sha256"],
            )
            target = expected_prediction_path(output_root, cell=cell, method_id=alias)
            if target.exists() or target.is_symlink():
                raise ExportError(f"refusing to overwrite Track A alias: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target_metadata = {
                "schema": PREDICTION_SCHEMA,
                "task_id": TASK_ID,
                "panel_sha256": panel_sha,
                "cell_id": cell.cell_id,
                "style": cell.style,
                "condition": cell.condition,
                "method_id": alias,
                "geometry_json": json.dumps(_geometry(cell), sort_keys=True),
                "binding_json": json.dumps(binding, sort_keys=True),
                "truth_opened": "false",
                "source_method_id": BASE_METHOD_ID,
                "source_iteration": str(iteration),
                "source_artifact_sha256": sha256_file(artifact_path),
                "source_evidence_sha256": sha256_file(evidence_path),
                "track_a_export_schema": EXPORT_SCHEMA,
                "serialization_only": "true",
            }
            save_file(
                {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
                str(target),
                metadata=target_metadata,
            )
            validate_prediction_artifact(
                target,
                cell=cell,
                panel_sha256=panel_sha,
                expected_method_id=alias,
                expected_binding=binding,
                repository_root=root,
            )
            method_bindings[cell.cell_id] = binding
            method_cells[cell.cell_id] = {
                "path": str(target.relative_to(root).as_posix()),
                "sha256": sha256_file(target),
                "bytes": int(target.stat().st_size),
                "source_artifact": _record_for_any_path(artifact_path, root=root),
                "source_evidence": _record_for_any_path(evidence_path, root=root),
                "source_iteration": iteration,
                "source_tensor_sha256": extra["tensor_sha256"],
            }
        # Store one exact binding because all cells share the same source
        # method state/code commitment except for the cell metadata in raw
        # source artifacts.  The per-cell artifact binding remains validated.
        manifest_methods[alias] = {
            "iteration": iteration,
            "binding_by_cell": method_bindings,
            "artifacts": method_cells,
        }
    manifest = {
        "schema": MAP_SCHEMA,
        "task_id": TASK_ID,
        "status": "SERIALIZATION_ONLY_EXPORT_COMPLETE",
        "created_utc": _now(),
        "panel": file_record(panel_path, repository_root=root),
        "diagnostic_root": str(diagnostic_root),
        "source_evidence": {
            cell_id: _record_for_any_path(evidence_path, root=root)
            for cell_id, (evidence_path, _evidence) in sorted(source_by_cell.items())
        },
        "method_ids": aliases,
        "methods": manifest_methods,
        "truth_opened": False,
        "source_tokens_read": False,
        "serialization_only": True,
        "exporter": file_record(adapter_path, repository_root=root),
        "exporter_execution_commit": commit,
        "exporter_commit_contains_file": _git_file_at_commit(root, adapter_path),
    }
    _json_create(output_root / "export_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output_root), "methods": aliases, "cells": len(cells), "truth_opened": False}, sort_keys=True))
    return 0


def _git_file_at_commit(root: Path, path: Path) -> bool:
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"HEAD:{path.relative_to(root).as_posix()}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, ValueError):
        return False
    return value.returncode == 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--panel", type=Path, default=Path("experiments/TRR-0003/footing/panel.json"))
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--iterations", type=int, nargs="+", default=list(DEFAULT_ITERATIONS))
    return parser


def main() -> int:
    try:
        return _export(_parser().parse_args())
    except (ExportError, FootingError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-0003 Track A export failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
