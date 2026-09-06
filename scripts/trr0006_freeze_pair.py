#!/usr/bin/env python3
"""Strict public freeze and pre-truth gate for the TRR-0006 prediction runner.

The gate consumes the runner's real artifacts: a frozen prediction
registration, the source-free observation manifest, ``predictions.json``,
``timings.json``, and ``run_manifest.json``.  It verifies the complete
four-cell by two-method matrix, rehashes every public/state/code artifact, and
loads each prediction safetensor to check keys, metadata, geometry, IDs, and
its tensor digest before a caller may open evaluator truth.

No truth path is opened, stat'ed, hashed, or parsed here.  The only tensors
loaded by this module are public observations (masks/positions) and public
prediction IDs.  The frozen receipt contains descriptors and hashes, never
truth or private labels.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import torch
from safetensors import safe_open

try:
    from scripts import trr0006_prediction_contract as contract
except ImportError:  # pragma: no cover - direct script/import fallback
    import trr0006_prediction_contract as contract


TASK_ID = contract.TASK_ID
SCHEMA = "token-reconstruction.trr0006-freeze-pair.v2"
RECEIPT_STATUS = "FROZEN_COMPLETE_PAIRED_MATRIX_NO_TRUTH"
CELL_ORDER = tuple(contract.CELL_ORDER)
METHOD_ORDER = tuple(contract.METHOD_IDS)
EXPECTED_KEYS = tuple((cell, method) for cell in CELL_ORDER for method in METHOD_ORDER)
EXPECTED_ENTRY_KEYS = tuple(f"{cell}::{method}" for cell, method in EXPECTED_KEYS)
SEQUENCE_TOKENS = contract.STORED_SEQUENCE_TOKENS
CAPTURE_TOKENS = contract.CAPTURE_SEQUENCE_TOKENS
POST_BOS_TOKENS = contract.SCORED_POST_BOS_TOKENS

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FreezePairError(ValueError):
    """Raised when a public TRR6 matrix cannot be frozen safely."""


class PretruthGateError(FreezePairError):
    """Raised when an executable public check fails before truth."""


def sha256_file(path: Path) -> str:
    return contract.sha256_file(Path(path))


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FreezePairError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezePairError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FreezePairError(f"{description} must be a JSON object")
    return dict(value)


def _resolve_root(value: Path | str) -> Path:
    root = Path(value).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise FreezePairError(f"repository root is unavailable: {root}")
    return root.resolve()


def _resolve_path(value: Path | str, *, root: Path, description: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise FreezePairError(f"{description} must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FreezePairError(f"{description} is unavailable: {candidate}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FreezePairError(f"{description} is not a regular file: {resolved}")
    return resolved


def _resolve_directory(value: Path | str, *, root: Path, description: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise FreezePairError(f"{description} must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FreezePairError(f"{description} is unavailable: {candidate}") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise FreezePairError(f"{description} is not a directory: {resolved}")
    return resolved


def _inside(path: Path, parent: Path, description: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise FreezePairError(f"{description} escaped required root: {path}") from exc


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _file_record(value: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    try:
        record = contract.validate_file_record(value, repository_root=root, description=description, verify=True)
    except contract.ContractError as exc:
        raise FreezePairError(str(exc)) from exc
    return record


def _actual_record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = _resolve_path(path, root=root, description=description)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def _same_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, root: Path, description: str) -> None:
    expected_path = _resolve_path(expected.get("path"), root=root, description=f"{description} expected path")
    actual_path = _resolve_path(actual.get("path"), root=root, description=f"{description} actual path")
    if actual_path != expected_path or int(actual.get("bytes", -1)) != int(expected.get("bytes", -2)) or actual.get("sha256") != expected.get("sha256"):
        raise FreezePairError(f"{description} binding changed")
    if actual_path.stat().st_size != int(expected["bytes"]) or sha256_file(actual_path) != expected["sha256"]:
        raise FreezePairError(f"{description} bytes/hash changed")


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreezePairError("cannot resolve current executable commit") from exc
    value = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise FreezePairError("current executable commit is not a full lowercase hash")
    return value


def _output_root(registration: Mapping[str, Any], root: Path) -> Path:
    raw = registration.get("output_root")
    if not isinstance(raw, str) or not raw:
        raise FreezePairError("registration output_root is absent")
    output = _resolve_directory(raw, root=root, description="prediction output root")
    task_root = (root / "experiments/TRR-0006").resolve()
    _inside(output, task_root, "prediction output root")
    return output


def _validate_code_and_runtime(registration: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    current_commit = _git_head(root)
    if current_commit != registration.get("code_commit"):
        raise FreezePairError(
            f"registration code commit differs from executable HEAD: {registration.get('code_commit')} != {current_commit}"
        )
    code_rows = registration.get("code_bindings")
    if not isinstance(code_rows, list) or not code_rows:
        raise FreezePairError("registration code bindings are incomplete")
    code_bindings = []
    for index, value in enumerate(code_rows):
        code_bindings.append(_file_record(value, root=root, description=f"code binding {index}"))
    runtime = registration.get("runtime_assets")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("normalized_public_E"), Mapping):
        raise FreezePairError("normalized public E binding is absent")
    embedding = _file_record(runtime["normalized_public_E"], root=root, description="normalized public E")
    return {"code_commit": current_commit, "code_bindings": code_bindings, "normalized_public_E": embedding}


def _load_registration(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _resolve_path(path, root=root, description="TRR6 registration")
    value = _load_json(path, "TRR6 registration")
    try:
        registration = contract.validate_registration(value)
    except contract.ContractError as exc:
        raise FreezePairError(str(exc)) from exc
    record = _actual_record(path, root=root, description="TRR6 registration")
    return registration, record


def _load_observations(
    registration: Mapping[str, Any],
    *,
    root: Path,
    supplied_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        manifest, parsed, manifest_record = contract.load_observation_manifest(
            registration,
            repository_root=root,
            verify_assets=True,
        )
    except contract.ContractError as exc:
        raise FreezePairError(str(exc)) from exc
    bound_path = _resolve_path(registration["observation_manifest"]["path"], root=root, description="observation manifest")
    if supplied_manifest_path is not None and _resolve_path(supplied_manifest_path, root=root, description="supplied observation manifest") != bound_path:
        raise FreezePairError("supplied observation manifest path differs from registration")
    cells: dict[str, dict[str, Any]] = {}
    for cell_id in CELL_ORDER:
        cell = parsed["cells"][cell_id]
        observation = cell["observation"]
        path = _resolve_path(observation["path"], root=root, description=f"observation {cell_id}")
        # Verify public geometry without materializing activation tensors.  The
        # masks and positions are small and safe to inspect before truth.
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                    raise FreezePairError(f"observation tensor keys changed: {cell_id}")
                h_shape = tuple(handle.get_slice("activations").get_shape())
                side_shape = tuple(handle.get_slice("attention_mask").get_shape())
                pos_shape = tuple(handle.get_slice("position_ids").get_shape())
                expected_h = (registration["records_per_domain"], SEQUENCE_TOKENS, contract.HIDDEN_SIZE)
                expected_side = (registration["records_per_domain"], SEQUENCE_TOKENS)
                if h_shape != expected_h or side_shape != expected_side or pos_shape != expected_side:
                    raise FreezePairError(f"observation geometry changed: {cell_id}")
                mask = handle.get_tensor("attention_mask").to(torch.bool).contiguous()
                positions = handle.get_tensor("position_ids").to(torch.long).contiguous()
        except FreezePairError:
            raise
        except Exception as exc:
            raise FreezePairError(f"observation artifact is unreadable: {cell_id}") from exc
        if not mask[:, 0].all().item() or (mask[:, 1:] > mask[:, :-1]).any().item():
            raise FreezePairError(f"observation mask is not BOS/right-padded: {cell_id}")
        # The registered estimand is an exactly 128-token clip and 127
        # post-BOS scores.  A shorter padded row would silently change it.
        if not mask.all().item():
            raise FreezePairError(f"observation clip is not complete at 127 post-BOS tokens: {cell_id}")
        expected_positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.long).repeat(mask.shape[0], 1)
        if not torch.equal(positions, expected_positions):
            raise FreezePairError(f"observation positions changed: {cell_id}")
        cells[cell_id] = {
            "cell_id": cell_id,
            "record_ids_sha256": cell["record_ids_sha256"],
            "observation": _actual_record(path, root=root, description=f"observation {cell_id}"),
        }
    return manifest, {"cells": cells, "records_per_domain": registration["records_per_domain"]}, manifest_record


def _state_binding(state: Mapping[str, Any], method_id: str, *, root: Path) -> dict[str, Any]:
    expected = contract.PUBLISHED_STATE_BINDINGS.get(method_id)
    if not isinstance(expected, Mapping):
        raise FreezePairError(f"unknown published method: {method_id}")
    if not isinstance(state, Mapping):
        raise FreezePairError(f"state binding is malformed: {method_id}")
    for key in ("bytes", "sha256", "source_commit"):
        if state.get(key) != expected[key]:
            raise FreezePairError(f"published state binding changed for {method_id}: {key}")
    actual_path = _resolve_path(state.get("path"), root=root, description=f"state {method_id}")
    expected_path = _resolve_path(expected["path"], root=root, description=f"registered state {method_id}")
    if actual_path != expected_path:
        raise FreezePairError(f"published state path changed for {method_id}")
    if actual_path.stat().st_size != expected["bytes"] or sha256_file(actual_path) != expected["sha256"]:
        raise FreezePairError(f"published state bytes/hash changed for {method_id}")
    return {"path": str(actual_path), "bytes": int(expected["bytes"]), "sha256": expected["sha256"], "source_commit": expected["source_commit"]}


def _prediction_tensor(path: Path, descriptor: Mapping[str, Any], *, registration: Mapping[str, Any], observation: Mapping[str, Any], root: Path) -> torch.Tensor:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"predictions"}:
                raise FreezePairError(f"prediction tensor keys changed: {descriptor.get('cell_id')}/{descriptor.get('method_id')}")
            metadata = dict(handle.metadata() or {})
            tensor = handle.get_tensor("predictions").to(torch.long).contiguous()
    except FreezePairError:
        raise
    except Exception as exc:
        raise FreezePairError(f"prediction artifact is unreadable: {path}") from exc
    records = int(registration["records_per_domain"])
    try:
        validated = contract.validate_prediction_tensor(tensor, records=records, sequence_tokens=SEQUENCE_TOKENS)
    except contract.ContractError as exc:
        raise FreezePairError(str(exc)) from exc
    required_metadata = {
        "schema": contract.PREDICTION_SCHEMA,
        "task_id": TASK_ID,
        "cell_id": descriptor.get("cell_id"),
        "method_id": descriptor.get("method_id"),
        "registration_sha256": descriptor.get("registration_sha256"),
        "observation_manifest_sha256": registration["observation_manifest"]["sha256"],
        "observation_sha256": observation["sha256"],
        "records": str(records),
        "sequence_tokens": str(SEQUENCE_TOKENS),
        "capture_sequence_tokens": str(CAPTURE_TOKENS),
        "hidden_size": str(contract.HIDDEN_SIZE),
        "candidate_arrays_persisted": "false",
        "truth_opened": "false",
    }
    for key, expected in required_metadata.items():
        if str(metadata.get(key)) != str(expected):
            raise FreezePairError(f"prediction metadata changed: {descriptor.get('cell_id')}/{descriptor.get('method_id')}/{key}")
    digest = contract.tensor_digest(validated)
    if digest != descriptor.get("prediction_sha256"):
        raise FreezePairError(f"prediction tensor digest changed: {descriptor.get('cell_id')}/{descriptor.get('method_id')}")
    return validated


def _validate_predictions_and_timings(
    registration: Mapping[str, Any],
    registration_record: Mapping[str, Any],
    observations: Mapping[str, Any],
    *,
    root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[tuple[str, str], torch.Tensor]]:
    prediction_manifest_path = output_root / "predictions.json"
    timing_manifest_path = output_root / "timings.json"
    run_manifest_path = output_root / "run_manifest.json"
    predictions_manifest = _load_json(prediction_manifest_path, "TRR6 prediction descriptor manifest")
    timings_manifest = _load_json(timing_manifest_path, "TRR6 timing descriptor manifest")
    run_manifest = _load_json(run_manifest_path, "TRR6 prediction run manifest")
    registration_sha = registration_record["sha256"]
    for value, status, description in (
        (predictions_manifest, "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH", "prediction descriptor manifest"),
        (timings_manifest, "PUBLIC_TIMINGS_COMPLETE_NO_TRUTH", "timing descriptor manifest"),
    ):
        if value.get("task_id") != TASK_ID or value.get("status") != status or value.get("truth_opened") is not False:
            raise FreezePairError(f"{description} is incomplete or truth-opened")
        if value.get("registration_sha256") != registration_sha:
            raise FreezePairError(f"{description} registration binding changed")
        if value.get("records_per_domain") != registration["records_per_domain"] or value.get("cell_order") != list(CELL_ORDER) or value.get("method_ids") != list(METHOD_ORDER):
            raise FreezePairError(f"{description} matrix binding changed")
    if run_manifest.get("schema") != contract.RUN_SCHEMA or run_manifest.get("task_id") != TASK_ID or run_manifest.get("status") != "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH":
        raise FreezePairError("prediction run manifest is incomplete")
    for key in ("truth_opened", "source_text_loaded", "target_labels_loaded", "candidate_arrays_persisted"):
        if run_manifest.get(key) is not False:
            raise FreezePairError(f"run manifest private/truth flag changed: {key}")
    if run_manifest.get("registration", {}).get("sha256") != registration_sha:
        raise FreezePairError("run manifest registration binding changed")
    if not run_manifest.get("predictions_complete") or not run_manifest.get("timing_decisions_complete"):
        raise FreezePairError("run manifest does not prove complete prediction/timing matrices")
    prediction_rows = predictions_manifest.get("predictions")
    timing_rows = timings_manifest.get("timings")
    if not isinstance(prediction_rows, Mapping) or not isinstance(timing_rows, Mapping):
        raise FreezePairError("prediction/timing descriptor maps are absent")
    if set(prediction_rows) != set(EXPECTED_ENTRY_KEYS) or set(timing_rows) != set(EXPECTED_ENTRY_KEYS):
        raise FreezePairError("prediction/timing descriptor matrix is incomplete")
    prediction_descriptors: dict[str, Any] = {}
    timing_descriptors: dict[str, Any] = {}
    prediction_tensors: dict[tuple[str, str], torch.Tensor] = {}
    for cell_id, method_id in EXPECTED_KEYS:
        entry_key = f"{cell_id}::{method_id}"
        descriptor = prediction_rows[entry_key]
        timing = timing_rows[entry_key]
        if not isinstance(descriptor, Mapping) or not isinstance(timing, Mapping):
            raise FreezePairError(f"descriptor is malformed: {entry_key}")
        if descriptor.get("schema") != contract.PREDICTION_SCHEMA or descriptor.get("task_id") != TASK_ID or descriptor.get("cell_id") != cell_id or descriptor.get("method_id") != method_id:
            raise FreezePairError(f"prediction descriptor binding changed: {entry_key}")
        if descriptor.get("truth_opened") is not False or descriptor.get("candidate_arrays_persisted") is not False:
            raise FreezePairError(f"prediction descriptor truth/candidate flag changed: {entry_key}")
        if descriptor.get("records") != registration["records_per_domain"] or descriptor.get("shape") != [registration["records_per_domain"], SEQUENCE_TOKENS]:
            raise FreezePairError(f"prediction descriptor geometry changed: {entry_key}")
        if descriptor.get("registration_sha256") != registration_sha:
            raise FreezePairError(f"prediction descriptor registration changed: {entry_key}")
        observation_binding = descriptor.get("observation")
        if not isinstance(observation_binding, Mapping):
            raise FreezePairError(f"prediction observation binding absent: {entry_key}")
        _same_record(observation_binding, observations["cells"][cell_id]["observation"], root=root, description=f"prediction observation {entry_key}")
        state = _state_binding(descriptor.get("state"), method_id, root=root)
        artifact_value = descriptor.get("prediction_artifact")
        artifact = _file_record(artifact_value, root=root, description=f"prediction artifact {entry_key}")
        artifact_path = _resolve_path(artifact["path"], root=root, description=f"prediction artifact {entry_key}")
        _inside(artifact_path, output_root, f"prediction artifact {entry_key}")
        tensor = _prediction_tensor(artifact_path, descriptor, registration=registration, observation=observation_binding, root=root)
        if descriptor.get("state", {}).get("sha256") != state["sha256"]:
            raise FreezePairError(f"prediction state digest changed: {entry_key}")
        if timing.get("schema") != contract.TIMING_SCHEMA or timing.get("task_id") != TASK_ID or timing.get("cell_id") != cell_id or timing.get("method_id") != method_id:
            raise FreezePairError(f"timing descriptor binding changed: {entry_key}")
        if timing.get("records") != registration["records_per_domain"] or timing.get("warmup_runs_per_record") != 1 or timing.get("measured_runs_per_record") != 1 or timing.get("warmup_output_exact_match_measured") is not True or timing.get("measured_output_selected") is not True:
            raise FreezePairError(f"timing repeat/coverage contract changed: {entry_key}")
        if timing.get("chunk_records") != contract.CAPTURE_BATCH_RECORDS or timing.get("chunks") != registration["records_per_domain"] // contract.CAPTURE_BATCH_RECORDS:
            raise FreezePairError(f"timing chunk contract changed: {entry_key}")
        per_record = timing.get("per_record_measured_seconds")
        if not isinstance(per_record, list) or len(per_record) != registration["records_per_domain"] or any(not isinstance(x, (int, float)) or not torch.isfinite(torch.tensor(float(x))).item() or float(x) <= 0 for x in per_record):
            raise FreezePairError(f"timing per-record coverage changed: {entry_key}")
        if timing.get("prediction_sha256") != descriptor.get("prediction_sha256"):
            raise FreezePairError(f"timing prediction digest changed: {entry_key}")
        if timing.get("prediction_artifact", {}).get("sha256") != artifact["sha256"]:
            raise FreezePairError(f"timing prediction artifact binding changed: {entry_key}")
        prediction_descriptors[entry_key] = {**dict(descriptor), "state": state, "prediction_artifact": artifact}
        timing_descriptors[entry_key] = dict(timing)
        prediction_tensors[(cell_id, method_id)] = tensor
    return prediction_descriptors, timing_descriptors, {
        "predictions": _actual_record(prediction_manifest_path, root=root, description="prediction descriptor manifest"),
        "timings": _actual_record(timing_manifest_path, root=root, description="timing descriptor manifest"),
        "run_manifest": _actual_record(run_manifest_path, root=root, description="prediction run manifest"),
    }, prediction_tensors


def _load_public_matrix(
    *,
    repository_root: Path | str,
    registration_path: Path | str,
    observation_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    root = _resolve_root(repository_root)
    registration, registration_record = _load_registration(Path(registration_path), root=root)
    executable = _validate_code_and_runtime(registration, root=root)
    output_root = _output_root(registration, root)
    manifest, observations, observation_record = _load_observations(
        registration,
        root=root,
        supplied_manifest_path=Path(observation_manifest_path) if observation_manifest_path is not None else None,
    )
    prediction_descriptors, timing_descriptors, output_records, prediction_tensors = _validate_predictions_and_timings(
        registration,
        registration_record,
        observations,
        root=root,
        output_root=output_root,
    )
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "repository_root": str(root),
        "output_root": str(output_root),
        "registration": registration,
        "registration_record": registration_record,
        "observation_manifest": manifest,
        "observation_record": observation_record,
        "observations": observations,
        "executable": executable,
        "prediction_descriptors": prediction_descriptors,
        "timing_descriptors": timing_descriptors,
        "output_records": output_records,
        "prediction_tensors": prediction_tensors,
        "truth_opened": False,
    }


def _optional_file_record(path: Path | str | None, *, root: Path, description: str) -> dict[str, Any] | None:
    if path is None:
        return None
    return _actual_record(_resolve_path(path, root=root, description=description), root=root, description=description)


def freeze_matrix(
    *,
    repository_root: Path | str,
    registration_path: Path | str,
    receipt_path: Path | str,
    observation_manifest_path: Path | str | None = None,
    plan_path: Path | str | None = None,
    panel_path: Path | str | None = None,
) -> dict[str, Any]:
    """Verify and create a complete paired receipt, without opening truth."""

    public = _load_public_matrix(
        repository_root=repository_root,
        registration_path=registration_path,
        observation_manifest_path=observation_manifest_path,
    )
    root = Path(public["repository_root"])
    registration_path_resolved = _resolve_path(registration_path, root=root, description="TRR6 registration")
    receipt_file = Path(receipt_path).expanduser()
    if not receipt_file.is_absolute():
        receipt_file = root / receipt_file
    receipt_file = receipt_file.resolve()
    _inside(receipt_file, root, "freeze receipt")
    if receipt_file.exists() or receipt_file.is_symlink():
        raise FreezePairError(f"freeze receipt already exists: {receipt_file}")
    plan_record = _optional_file_record(plan_path, root=root, description="decision plan")
    panel_record = _optional_file_record(panel_path, root=root, description="source panel")
    receipt = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": RECEIPT_STATUS,
        "truth_opened": False,
        "repository_root": str(root),
        "output_root": _relative(Path(public["output_root"]), root),
        "registration": public["registration_record"],
        "observation_manifest": public["observation_record"],
        "plan": plan_record,
        "panel": panel_record,
        "code_commit": public["executable"]["code_commit"],
        "executable": public["executable"],
        "matrix": {
            "cells": list(CELL_ORDER),
            "methods": list(METHOD_ORDER),
            "entries": len(EXPECTED_KEYS),
            "records_per_domain": public["registration"]["records_per_domain"],
            "capture_sequence_tokens": CAPTURE_TOKENS,
            "stored_sequence_tokens": SEQUENCE_TOKENS,
            "scored_post_bos_tokens": POST_BOS_TOKENS,
        },
        "output_records": public["output_records"],
        "entries": [
            {
                "entry_key": key,
                "prediction": public["prediction_descriptors"][key],
                "timing": public["timing_descriptors"][key],
            }
            for key in EXPECTED_ENTRY_KEYS
        ],
    }
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt_file.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise FreezePairError(f"freeze receipt already exists: {receipt_file}") from exc
    return receipt


def _check_optional_binding(receipt_value: Any, supplied: Path | str | None, *, root: Path, description: str) -> None:
    if receipt_value is None:
        if supplied is not None:
            raise PretruthGateError(f"{description} was supplied but not frozen")
        return
    if not isinstance(receipt_value, Mapping):
        raise PretruthGateError(f"{description} receipt binding is malformed")
    if supplied is None:
        path = receipt_value.get("path")
    else:
        path = supplied
    actual = _actual_record(_resolve_path(path, root=root, description=description), root=root, description=description)
    _same_record(actual, receipt_value, root=root, description=description)


def validate_before_truth(
    *,
    repository_root: Path | str,
    registration_path: Path | str,
    receipt_path: Path | str,
    observation_manifest_path: Path | str | None = None,
    plan_path: Path | str | None = None,
    panel_path: Path | str | None = None,
    truth_path: Path | str | None = None,
) -> dict[str, Any]:
    """Revalidate every public artifact before a caller invokes truth."""

    root = _resolve_root(repository_root)
    receipt_file = Path(receipt_path).expanduser()
    if not receipt_file.is_absolute():
        receipt_file = root / receipt_file
    receipt_file = receipt_file.resolve()
    receipt = _load_json(receipt_file, "TRR6 freeze receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("task_id") != TASK_ID or receipt.get("status") != RECEIPT_STATUS or receipt.get("truth_opened") is not False:
        raise PretruthGateError("freeze receipt is not a complete closed matrix")
    _inside(receipt_file, root, "freeze receipt")
    public = _load_public_matrix(
        repository_root=root,
        registration_path=registration_path,
        observation_manifest_path=observation_manifest_path,
    )
    if receipt.get("code_commit") != public["executable"]["code_commit"]:
        raise PretruthGateError("freeze receipt executable commit changed")
    for role, actual in (
        ("registration", public["registration_record"]),
        ("observation_manifest", public["observation_record"]),
    ):
        bound = receipt.get(role)
        if not isinstance(bound, Mapping):
            raise PretruthGateError(f"freeze receipt {role} binding is absent")
        _same_record(actual, bound, root=root, description=role)
    if receipt.get("executable") != public["executable"]:
        raise PretruthGateError("freeze receipt executable artifact bindings changed")
    bound_outputs = receipt.get("output_records")
    if not isinstance(bound_outputs, Mapping):
        raise PretruthGateError("freeze receipt output artifact bindings are absent")
    for role in ("predictions", "timings", "run_manifest"):
        bound = bound_outputs.get(role)
        actual = public["output_records"].get(role)
        if not isinstance(bound, Mapping) or not isinstance(actual, Mapping):
            raise PretruthGateError(f"freeze receipt {role} output binding is absent")
        _same_record(actual, bound, root=root, description=f"{role} output")
    _check_optional_binding(receipt.get("plan"), plan_path, root=root, description="decision plan")
    _check_optional_binding(receipt.get("panel"), panel_path, root=root, description="source panel")
    if receipt.get("output_root") != _relative(Path(public["output_root"]), root):
        raise PretruthGateError("freeze receipt output root changed")
    matrix = receipt.get("matrix")
    if not isinstance(matrix, Mapping) or matrix.get("cells") != list(CELL_ORDER) or matrix.get("methods") != list(METHOD_ORDER) or matrix.get("entries") != len(EXPECTED_KEYS) or matrix.get("records_per_domain") != public["registration"]["records_per_domain"]:
        raise PretruthGateError("freeze receipt matrix binding changed")
    receipt_entries = receipt.get("entries")
    if not isinstance(receipt_entries, list) or len(receipt_entries) != len(EXPECTED_KEYS):
        raise PretruthGateError("freeze receipt has a partial matrix")
    by_key = {entry.get("entry_key"): entry for entry in receipt_entries if isinstance(entry, Mapping)}
    if set(by_key) != set(EXPECTED_ENTRY_KEYS):
        raise PretruthGateError("freeze receipt entry matrix is incomplete")
    for key in EXPECTED_ENTRY_KEYS:
        entry = by_key[key]
        if entry.get("prediction") != public["prediction_descriptors"][key] or entry.get("timing") != public["timing_descriptors"][key]:
            raise PretruthGateError(f"public descriptor changed after freeze: {key}")
    # Do not inspect truth content.  This path check is only to prevent a
    # private sidecar from being placed under the frozen public output root.
    if truth_path is not None:
        candidate = Path(truth_path).expanduser().absolute().resolve(strict=False)
        try:
            candidate.relative_to(Path(public["output_root"]))
        except ValueError:
            pass
        else:
            raise PretruthGateError("truth path is inside the frozen prediction root")
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_MATRIX_VERIFIED_BEFORE_TRUTH",
        "verified_before_truth": True,
        "truth_opened": False,
        "receipt_path": _relative(receipt_file, root),
        "records_per_domain": public["registration"]["records_per_domain"],
        "entry_count": len(EXPECTED_KEYS),
        "cells": list(CELL_ORDER),
        "methods": list(METHOD_ORDER),
        "registration": public["registration"],
        "observation_manifest": public["observation_manifest"],
        "observations": public["observations"],
        "prediction_descriptors": public["prediction_descriptors"],
        "timing_descriptors": public["timing_descriptors"],
        "prediction_tensors": public["prediction_tensors"],
        "assets_rehashed": True,
        "code_commit": public["executable"]["code_commit"],
    }


__all__ = [
    "CAPTURE_TOKENS",
    "CELL_ORDER",
    "EXPECTED_ENTRY_KEYS",
    "EXPECTED_KEYS",
    "FreezePairError",
    "METHOD_ORDER",
    "POST_BOS_TOKENS",
    "PretruthGateError",
    "RECEIPT_STATUS",
    "SCHEMA",
    "SEQUENCE_TOKENS",
    "freeze_matrix",
    "sha256_file",
    "sha256_json",
    "validate_before_truth",
]
