#!/usr/bin/env python3
"""Build the 13-method TRR-0003 exploratory bundle registration.

The registry is created only after Track A's serialization aliases, Track B's
standalone decoders, and the three historical comparators have all written
complete public prediction bundles.  This process validates every artifact
and binding but does not open a truth sidecar.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from token_reconstruction.footing import (
    REGISTRATION_SCHEMA,
    TASK_ID,
    FootingError,
    expected_cell_ids,
    expected_prediction_path,
    file_record,
    load_all_cells,
    load_method_registry,
    load_panel,
    sha256_file,
    validate_prediction_artifact,
)


DEFAULT_PANEL = Path("experiments/TRR-0003/footing/panel.json")
DEFAULT_TRUTH_BINDING = Path("experiments/TRR-0003/footing/truth_binding_v2.json")
DEFAULT_TRACK_A = Path("outputs/TRR-0003/track_a_export_v1")
DEFAULT_TRACK_B = Path("outputs/TRR-0003/track_b/panel_selected_v1")
DEFAULT_COMPARATOR = Path("outputs/TRR-0003/comparator_v1")
DEFAULT_OUTPUT = Path("experiments/TRR-0003/footing/method_registry.json")
TRACK_A_IDS = tuple(
    f"checkpoint_reverse_fixed_point_euclidean_k{16}_i{iteration:03d}"
    for iteration in (0, 1, 2, 4, 8, 16, 32)
)
TRACK_B_IDS = (
    "angular_inverse_control",
    "tied_affine_token_ce",
    "residual_mlp256_token_ce",
)
COMPARATOR_IDS = (
    "historical_alpaca_a1",
    "frozen_a1_a2_k256",
    "direct_inverse",
)


class RegistryError(RuntimeError):
    """Raised when one registered bundle is missing or inconsistent."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RegistryError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"{description} root is not an object: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return (root / path if not path.is_absolute() else path).resolve()


def _create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RegistryError(f"refusing to overwrite method registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()


def _require_canonical_root(root: Path, bundle: Path, *, description: str) -> str:
    if bundle.is_symlink() or not bundle.is_dir():
        raise RegistryError(f"{description} bundle root is unavailable: {bundle}")
    try:
        relative = bundle.relative_to(root).as_posix()
    except ValueError as exc:
        raise RegistryError(f"{description} bundle root is outside repository: {bundle}") from exc
    if not relative or relative.startswith("../"):
        raise RegistryError(f"{description} bundle root path is unsafe")
    return relative


def _binding_from_file(path: Path, *, method_ids: tuple[str, ...]) -> dict[str, Mapping[str, Any]]:
    payload = _load_json(path, description="prediction binding manifest")
    if set(payload) != set(method_ids):
        raise RegistryError(f"prediction binding set is incomplete: {path}")
    result: dict[str, Mapping[str, Any]] = {}
    for method_id in method_ids:
        value = payload.get(method_id)
        if not isinstance(value, Mapping):
            raise RegistryError(f"prediction binding is malformed: {method_id}")
        result[method_id] = dict(value)
    return result


def _a_bindings(
    *, root: Path, panel: Mapping[str, Any], panel_path: Path, bundle: Path
) -> dict[str, Mapping[str, Any]]:
    manifest_path = bundle / "export_manifest.json"
    manifest = _load_json(manifest_path, description="Track A export manifest")
    if manifest.get("schema") != "token-reconstruction.trr0003-track-a-export-manifest.v1":
        raise RegistryError("Track A export manifest schema changed")
    if manifest.get("task_id") != TASK_ID or manifest.get("truth_opened") is not False:
        raise RegistryError("Track A export manifest truth contract changed")
    if manifest.get("method_ids") != list(TRACK_A_IDS):
        raise RegistryError("Track A alias method order changed")
    cells = load_all_cells(panel, repository_root=root)
    panel_sha = sha256_file(panel_path)
    result: dict[str, Mapping[str, Any]] = {}
    methods = manifest.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(TRACK_A_IDS):
        raise RegistryError("Track A export method set is incomplete")
    for method_id in TRACK_A_IDS:
        row = methods.get(method_id)
        if not isinstance(row, Mapping):
            raise RegistryError(f"Track A export method row is malformed: {method_id}")
        bindings = row.get("binding_by_cell")
        if not isinstance(bindings, Mapping) or set(bindings) != set(expected_cell_ids()):
            raise RegistryError(f"Track A binding cells are incomplete: {method_id}")
        serialized = [json.dumps(bindings[cell_id], sort_keys=True) for cell_id in expected_cell_ids()]
        if len(set(serialized)) != 1:
            raise RegistryError(f"Track A alias bindings differ across cells: {method_id}")
        binding = bindings[expected_cell_ids()[0]]
        if not isinstance(binding, Mapping):
            raise RegistryError(f"Track A alias binding is malformed: {method_id}")
        result[method_id] = dict(binding)
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_cell_ids()):
            raise RegistryError(f"Track A artifact cells are incomplete: {method_id}")
        for cell in cells:
            target = expected_prediction_path(bundle, cell=cell, method_id=method_id)
            validate_prediction_artifact(
                target,
                cell=cell,
                panel_sha256=panel_sha,
                expected_method_id=method_id,
                expected_binding=binding,
                repository_root=root,
            )
            artifact = artifacts.get(cell.cell_id)
            if not isinstance(artifact, Mapping) or artifact.get("sha256") != sha256_file(target):
                raise RegistryError(f"Track A artifact manifest hash changed: {target}")
    return result


def _standard_bindings(
    *, root: Path, panel: Mapping[str, Any], panel_path: Path, bundle: Path, method_ids: tuple[str, ...], binding_file: str
) -> dict[str, Mapping[str, Any]]:
    bindings = _binding_from_file(bundle / binding_file, method_ids=method_ids)
    cells = load_all_cells(panel, repository_root=root)
    panel_sha = sha256_file(panel_path)
    for method_id in method_ids:
        binding = bindings[method_id]
        serialized = json.dumps(binding, sort_keys=True)
        for cell in cells:
            target = expected_prediction_path(bundle, cell=cell, method_id=method_id)
            validate_prediction_artifact(
                target,
                cell=cell,
                panel_sha256=panel_sha,
                expected_method_id=method_id,
                expected_binding=binding,
                repository_root=root,
            )
        # Ensure the source bundle really has one fixed binding per method; the
        # common scorer has one expected binding map for each registered ID.
        if not serialized:
            raise RegistryError(f"empty binding: {method_id}")
    return bindings


def build(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    panel_path = _resolve(root, args.panel)
    panel = load_panel(panel_path, repository_root=root)
    truth_binding_path = _resolve(root, args.truth_binding)
    truth_binding = _load_json(truth_binding_path, description="truth binding")
    if truth_binding.get("task_id") != TASK_ID:
        raise RegistryError("truth binding task changed")
    track_a = _resolve(root, args.track_a)
    track_b = _resolve(root, args.track_b)
    comparator = _resolve(root, args.comparator)
    a_bindings = _a_bindings(root=root, panel=panel, panel_path=panel_path, bundle=track_a)
    b_bindings = _standard_bindings(
        root=root, panel=panel, panel_path=panel_path, bundle=track_b, method_ids=TRACK_B_IDS, binding_file="bindings.json"
    )
    c_bindings = _standard_bindings(
        root=root, panel=panel, panel_path=panel_path, bundle=comparator, method_ids=COMPARATOR_IDS, binding_file="bindings.json"
    )
    rows: list[dict[str, Any]] = []
    for method_id in TRACK_A_IDS:
        rows.append({
            "id": method_id,
            "track": "track_a",
            "candidate_policy": "required",
            "binding": a_bindings[method_id],
            "bundle": {"layout": "canonical", "root": _require_canonical_root(root, track_a, description="Track A")},
        })
    for method_id in TRACK_B_IDS:
        rows.append({
            "id": method_id,
            "track": "track_b",
            "candidate_policy": "forbidden",
            "binding": b_bindings[method_id],
            "bundle": {"layout": "canonical", "root": _require_canonical_root(root, track_b, description="Track B")},
        })
    for method_id in COMPARATOR_IDS:
        rows.append({
            "id": method_id,
            "track": "comparator",
            "candidate_policy": "required",
            "binding": c_bindings[method_id],
            "bundle": {"layout": "canonical", "root": _require_canonical_root(root, comparator, description="comparator")},
        })
    methods = [row["id"] for row in rows]
    payload = {
        "schema": REGISTRATION_SCHEMA,
        "task_id": TASK_ID,
        "status": "EXPLORATORY_METHOD_BUNDLE_REGISTRATION",
        "created_utc": _now(),
        "scope": "TRR-0003 shared retrospective development panel; pilot diagnostic only",
        "canonical_setups": {"track_a": "NOT_RUN", "track_b": "NOT_RUN", "overall": "INCOMPLETE"},
        "panel": file_record(panel_path, repository_root=root),
        "truth_binding": truth_binding,
        "method_ids": methods,
        "methods": rows,
        "source_bundles": {
            "track_a_export_manifest": file_record(track_a / "export_manifest.json", repository_root=root),
            "track_b_bindings": file_record(track_b / "bindings.json", repository_root=root),
            "comparator_bindings": file_record(comparator / "bindings.json", repository_root=root),
        },
    }
    output = _resolve(root, args.output)
    _create(output, payload)
    # Re-read through the production loader so registry path, binding files,
    # bundle roots, truth commitment, and all 13 method rows are exercised now.
    loaded = load_method_registry(
        output,
        repository_root=root,
        panel=panel,
        panel_path=panel_path,
        require_truth_binding=True,
    )
    if tuple(loaded["method_ids"]) != tuple(methods):
        raise RegistryError("production registry loader changed method order")
    print(json.dumps({"status": payload["status"], "methods": methods, "output": str(output), "truth_opened": False}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--truth-binding", type=Path, default=DEFAULT_TRUTH_BINDING)
    parser.add_argument("--track-a", type=Path, default=DEFAULT_TRACK_A)
    parser.add_argument("--track-b", type=Path, default=DEFAULT_TRACK_B)
    parser.add_argument("--comparator", type=Path, default=DEFAULT_COMPARATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    try:
        return build(_parser().parse_args())
    except (RegistryError, FootingError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-0003 registry build failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
