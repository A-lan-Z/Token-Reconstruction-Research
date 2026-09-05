#!/usr/bin/env python3
"""Check public-prefix arithmetic on a disjoint public validation slice.

This is a matched-model diagnostic for the Track A checkpoint-only pilot.  It
loads the public auxiliary validation token slice only after checking the
preparation evidence and then compares a forward pass through the actually
loaded public prefix with the cached boundary activations.  It does not load
the shared panel, evaluator truth, a tokenizer, or any fitted state, and it
does not produce a reconstruction prediction.  The result is therefore a
control for cache/model arithmetic rather than evidence of unknown-target
recovery.

The loader and public resource binding are reused from
``trr0003_track_a.py`` so this control exercises the same BF16 hashing,
checkpoint, prefix, and deterministic-SDPA paths as the deployed runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource as sys_resource
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch

from token_reconstruction.footing import (
    BOS_TOKEN_ID,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    TASK_ID,
    file_record,
)

import trr0003_track_a as track_a


IDENTITY_SCHEMA = "token-reconstruction.trr0003-track-a-public-forward-identity.v1"
VALIDATION_SCHEMA = "token-reconstruction.trr0003-track-b-public-validation-slice.v1"
VALIDATION_RECORD_SCHEMA = "token-reconstruction.trr0003-track-b-public-validation-records.v1"
VALIDATION_EVIDENCE_SCHEMA = (
    "token-reconstruction.trr0003-track-b-public-validation-preparation.v1"
)
VOCAB_SIZE = 128256
DEFAULT_BATCH_SIZE = 8
DEFAULT_MIN_FREE_BYTES = track_a.DEFAULT_MIN_FREE_BYTES
DEFAULT_PROBE_BYTES = track_a.DEFAULT_PROBE_BYTES
DEFAULT_MAX_SECONDS = 1200.0


class IdentityControlError(RuntimeError):
    """Raised when the public validation identity control cannot run safely."""


@dataclass(frozen=True)
class ValidationAssets:
    observation_path: Path
    truth_path: Path
    records_path: Path
    evidence_path: Path
    observation: torch.Tensor
    truth: torch.Tensor
    record_ids: tuple[str, ...]
    observation_metadata: dict[str, str]
    truth_metadata: dict[str, str]
    records_sha256: str
    evidence_sha256: str


def _json_load(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise IdentityControlError(f"JSON input must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityControlError(f"invalid JSON input: {path}") from exc


def _write_json_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise IdentityControlError(f"refusing to overwrite identity artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise IdentityControlError(f"refusing to overwrite identity artifact: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _regular_file(path: Path, *, description: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise IdentityControlError(f"{description} must be a regular file: {path}")
    return path


def _evidence_output_path(
    evidence: Mapping[str, Any], *, key: str, repository_root: Path, expected: Path
) -> tuple[Path, dict[str, Any]]:
    outputs = evidence.get("outputs")
    if not isinstance(outputs, Mapping):
        raise IdentityControlError("validation evidence outputs are absent")
    row = outputs.get(key)
    if not isinstance(row, Mapping):
        raise IdentityControlError(f"validation evidence output is absent: {key}")
    relative = row.get("path")
    if not isinstance(relative, str) or not relative:
        raise IdentityControlError(f"validation evidence output path is absent: {key}")
    actual = (repository_root / relative).resolve()
    expected_resolved = expected.resolve()
    if actual != expected_resolved:
        raise IdentityControlError(
            f"validation evidence {key} path is not the requested slice: {actual}"
        )
    _regular_file(actual, description=f"validation {key} output")
    expected_bytes = row.get("bytes")
    expected_sha = row.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise IdentityControlError(f"validation evidence byte count is invalid: {key}")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise IdentityControlError(f"validation evidence hash is invalid: {key}")
    if actual.stat().st_size != expected_bytes or track_a.sha256_file(actual) != expected_sha:
        raise IdentityControlError(f"validation evidence output changed: {key}")
    return actual, dict(row)


def _check_disjoint_evidence(
    evidence_path: Path, *, repository_root: Path, validation_root: Path
) -> dict[str, Any]:
    evidence = _json_load(evidence_path)
    if not isinstance(evidence, Mapping):
        raise IdentityControlError("validation evidence is not an object")
    if evidence.get("schema") != VALIDATION_EVIDENCE_SCHEMA:
        raise IdentityControlError("validation evidence schema changed")
    if evidence.get("task_id") != TASK_ID or evidence.get("track") != "track_b":
        raise IdentityControlError("validation evidence identity changed")
    disjointness = evidence.get("disjointness")
    if not isinstance(disjointness, Mapping):
        raise IdentityControlError("validation disjointness evidence is absent")
    if disjointness.get("checked_before_validation_label_access") is not True:
        raise IdentityControlError("validation labels were not gated by disjointness checks")
    overlaps = disjointness.get("overlap_counts")
    if not isinstance(overlaps, Mapping) or set(overlaps) != {
        "panel",
        "inverse_train",
        "target_update_train",
        "blind_evaluation",
    }:
        raise IdentityControlError("validation overlap evidence is incomplete")
    if any(value != 0 for value in overlaps.values()):
        raise IdentityControlError(f"public validation slice overlaps a reserved split: {overlaps}")
    selection = evidence.get("selection")
    if not isinstance(selection, Mapping) or selection.get("split") != "development":
        raise IdentityControlError("validation slice is not the declared public development split")
    row_slice = selection.get("row_slice")
    if row_slice != {"start": 8, "stop": 32, "stop_is_exclusive": True}:
        raise IdentityControlError("validation row slice changed from the disjoint [8:32) slice")
    if validation_root.resolve() != (repository_root / "outputs/TRR-0003/track_b/public_validation_slice_v2").resolve():
        raise IdentityControlError("identity control requires the registered v2 validation slice")
    # Resolve and hash only the derived validation outputs after the checks
    # above.  In particular, this function never opens the source truth file
    # named in the preparation receipt.
    for key, expected_name in (
        ("observations", "public_validation_observations.safetensors"),
        ("records", "public_validation_records.json"),
        ("truth", "public_validation_truth.safetensors"),
    ):
        _evidence_output_path(
            evidence,
            key=key,
            repository_root=repository_root,
            expected=validation_root / expected_name,
        )
    return {
        "schema": evidence["schema"],
        "task_id": evidence["task_id"],
        "selection": dict(selection),
        "overlap_counts": {str(key): int(value) for key, value in overlaps.items()},
        "checked_before_validation_label_access": True,
        "source_observations": dict(evidence.get("source_observations", {})),
        "source_records": dict(evidence.get("source_records", {})),
        "preparation_git_commit": evidence.get("git_commit_at_start_and_end"),
    }


def _load_validation_observation(path: Path) -> tuple[torch.Tensor, dict[str, str]]:
    """Load only the activation tensor from the derived public slice."""

    _regular_file(path, description="public validation observations")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations"}:
                raise IdentityControlError("validation observations contain unexpected tensor fields")
            metadata = handle.metadata() or {}
            if not isinstance(metadata, Mapping):
                raise IdentityControlError("validation observation metadata is malformed")
            metadata = {str(key): str(value) for key, value in metadata.items()}
            if metadata.get("schema") != VALIDATION_SCHEMA:
                raise IdentityControlError("validation observation schema changed")
            if metadata.get("truth_source_not_included") != "false":
                raise IdentityControlError("validation observation does not assert source truth exclusion")
            value = handle.get_tensor("activations").contiguous()
    except IdentityControlError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise IdentityControlError(f"public validation observations are unreadable: {path}") from exc
    if tuple(value.shape) != (24, 40, HIDDEN_SIZE):
        raise IdentityControlError(f"validation observation geometry changed: {tuple(value.shape)}")
    if not value.dtype.is_floating_point or not torch.isfinite(value).all().item():
        raise IdentityControlError("validation observations are not finite floating point")
    return value, metadata


def _load_validation_truth(path: Path) -> tuple[torch.Tensor, dict[str, str]]:
    """Load public auxiliary token IDs after the disjointness gate."""

    _regular_file(path, description="public validation labels")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"token_ids"}:
                raise IdentityControlError("validation labels contain unexpected tensor fields")
            metadata = handle.metadata() or {}
            if not isinstance(metadata, Mapping):
                raise IdentityControlError("validation label metadata is malformed")
            metadata = {str(key): str(value) for key, value in metadata.items()}
            if metadata.get("schema") != (
                "token-reconstruction.trr0003-track-b-public-validation-label-slice.v1"
            ):
                raise IdentityControlError("validation label schema changed")
            if metadata.get("truth_role") != "public auxiliary validation only":
                raise IdentityControlError("validation labels are not marked public auxiliary")
            value = handle.get_tensor("token_ids").contiguous()
    except IdentityControlError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise IdentityControlError(f"public validation labels are unreadable: {path}") from exc
    if tuple(value.shape) != (24, 40):
        raise IdentityControlError(f"validation label geometry changed: {tuple(value.shape)}")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise IdentityControlError("validation labels are not integer token IDs")
    if value[:, 0].ne(BOS_TOKEN_ID).any().item() or value.lt(0).any().item() or value.ge(VOCAB_SIZE).any().item():
        raise IdentityControlError("validation labels contain an invalid token ID")
    return value, metadata


def _load_validation_records(path: Path) -> tuple[str, ...]:
    raw = _json_load(path)
    if not isinstance(raw, Mapping) or raw.get("schema") != VALIDATION_RECORD_SCHEMA:
        raise IdentityControlError("validation record schema changed")
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != 24:
        raise IdentityControlError("validation record count changed")
    record_ids: list[str] = []
    source_rows: list[int] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("record_id"), str):
            raise IdentityControlError("validation record ID is malformed")
        if not isinstance(record.get("source_row"), int):
            raise IdentityControlError("validation source row is malformed")
        record_ids.append(record["record_id"])
        source_rows.append(record["source_row"])
    if len(set(record_ids)) != 24 or source_rows != list(range(8, 32)):
        raise IdentityControlError("validation record ordering changed")
    return tuple(record_ids)


def _load_validation_assets(
    *, repository_root: Path, validation_root: Path, evidence_path: Path
) -> ValidationAssets:
    evidence_summary = _check_disjoint_evidence(
        evidence_path, repository_root=repository_root, validation_root=validation_root
    )
    observation_path = validation_root / "public_validation_observations.safetensors"
    records_path = validation_root / "public_validation_records.json"
    truth_path = validation_root / "public_validation_truth.safetensors"
    observation, observation_metadata = _load_validation_observation(observation_path)
    record_ids = _load_validation_records(records_path)
    # Labels are opened only after _check_disjoint_evidence has completed.
    truth, truth_metadata = _load_validation_truth(truth_path)
    if observation.shape[:2] != truth.shape or len(record_ids) != observation.shape[0]:
        raise IdentityControlError("validation assets disagree on record geometry")
    return ValidationAssets(
        observation_path=observation_path,
        truth_path=truth_path,
        records_path=records_path,
        evidence_path=evidence_path,
        observation=observation,
        truth=truth,
        record_ids=record_ids,
        observation_metadata=observation_metadata,
        truth_metadata=truth_metadata,
        records_sha256=track_a.sha256_file(records_path),
        evidence_sha256=track_a.sha256_file(evidence_path),
    )


def _metrics(actual: torch.Tensor, cached: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    cached_f = cached.float()
    error = actual_f - cached_f
    reference_norm = torch.linalg.vector_norm(cached_f.reshape(-1)).clamp_min(1e-12)
    error_norm = torch.linalg.vector_norm(error.reshape(-1))
    return {
        "max_abs": float(error.abs().max().item()),
        "mean_abs": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "relative_l2": float((error_norm / reference_norm).item()),
        "allclose_atol_1e-3_rtol_1e-3": bool(torch.allclose(actual_f, cached_f, atol=1e-3, rtol=1e-3)),
        "allclose_atol_1e-2_rtol_1e-2": bool(torch.allclose(actual_f, cached_f, atol=1e-2, rtol=1e-2)),
    }


def _code_binding(repository_root: Path, resource: track_a.ResourceBundle) -> dict[str, Any]:
    code_paths = (
        Path(__file__).resolve(),
        Path(track_a.__file__).resolve(),
        repository_root / "src/token_reconstruction/checkpoint_inverse.py",
        repository_root / "src/token_reconstruction/public_prefix.py",
        repository_root / "src/token_reconstruction/footing.py",
    )
    code = [track_a._repo_record(path, repository_root) for path in code_paths]
    commit = track_a._git_head()
    if commit is None:
        raise IdentityControlError("identity control requires a full Git commit")
    return {
        "code": code,
        "code_commit": commit,
        "resource_manifest": track_a._resource_binding(resource),
    }


def _identity_binding(
    *,
    repository_root: Path,
    assets: ValidationAssets,
    resource: track_a.ResourceBundle,
    state: track_a.LoadedPublicState,
) -> dict[str, Any]:
    result = _code_binding(repository_root, resource)
    result.update(
        {
            "validation_evidence": track_a._repo_record(
                assets.evidence_path, repository_root
            ),
            "validation_observations": track_a._repo_record(
                assets.observation_path, repository_root
            ),
            "validation_labels": track_a._repo_record(assets.truth_path, repository_root),
            "validation_records": track_a._repo_record(assets.records_path, repository_root),
            "loaded_public_prefix": {
                "state_sha256": state.prefix_digest,
                "embedding_sha256": state.embedding_digest,
                "parameter_bytes": state.parameter_bytes,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            },
            "fixed_control": {
                "cut_depth": CUT_DEPTH,
                "forward": "public_prefix.forward_full(token_ids)",
                "position_ids": "arange(valid_length)",
                "attention": "full causal mask; deterministic SDPA math",
                "batch_size": DEFAULT_BATCH_SIZE,
                "truth_role": "public auxiliary labels generate a matched diagnostic input only",
                "deployed_reconstruction": False,
                "candidate_prefix_simulations": 0,
                "fitted_parameters": False,
            },
        }
    )
    return result


def _forward_control(
    *,
    assets: ValidationAssets,
    state: track_a.LoadedPublicState,
    guard: track_a.ResourceGuard,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, assets.truth.shape[0], DEFAULT_BATCH_SIZE):
            guard.check(f"public validation forward batch {start}")
            stop = min(assets.truth.shape[0], start + DEFAULT_BATCH_SIZE)
            input_ids = assets.truth[start:stop].to(device="cuda", dtype=torch.long)
            cached = assets.observation[start:stop].to(device="cuda")
            batch_started = time.perf_counter()
            actual = state.prefix.forward_full(input_ids)
            if tuple(actual.shape) != tuple(cached.shape):
                raise IdentityControlError("public-prefix forward geometry differs from cached H4")
            if not torch.isfinite(actual).all().item():
                raise IdentityControlError("public-prefix forward produced non-finite H4")
            track_a.synchronize()
            batch_seconds = time.perf_counter() - batch_started
            actual_cpu = actual.detach().to(device="cpu", dtype=torch.float32)
            cached_cpu = cached.detach().to(device="cpu", dtype=torch.float32)
            for offset in range(stop - start):
                metric = _metrics(actual_cpu[offset], cached_cpu[offset])
                metric.update(
                    {
                        "record_index": start + offset,
                        "record_id": assets.record_ids[start + offset],
                        "valid_tokens": int(actual_cpu.shape[1]),
                        "forward_seconds_batch": batch_seconds,
                    }
                )
                rows.append(metric)
            del input_ids, cached, actual, actual_cpu, cached_cpu
            track_a.synchronize()
    all_error = {
        "max_abs": max(float(row["max_abs"]) for row in rows),
        "mean_abs": sum(float(row["mean_abs"]) for row in rows) / len(rows),
        "rmse": math.sqrt(
            sum(float(row["rmse"]) ** 2 for row in rows) / len(rows)
        ),
        "relative_l2_mean": sum(float(row["relative_l2"]) for row in rows) / len(rows),
        "allclose_atol_1e-3_rtol_1e-3": all(
            bool(row["allclose_atol_1e-3_rtol_1e-3"]) for row in rows
        ),
        "allclose_atol_1e-2_rtol_1e-2": all(
            bool(row["allclose_atol_1e-2_rtol_1e-2"]) for row in rows
        ),
        "records": len(rows),
        "forward_seconds": time.perf_counter() - started,
        "public_prefix_layer_evaluations": len(rows) * CUT_DEPTH,
        "candidate_prefix_simulations": 0,
    }
    return rows, all_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/public_validation_slice_v2"),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--probe-bytes", type=int, default=DEFAULT_PROBE_BYTES)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    validation_root = args.validation_root.resolve()
    output_path = args.output.resolve()
    if args.batch_size != DEFAULT_BATCH_SIZE:
        raise IdentityControlError(
            f"identity control batch size is frozen at {DEFAULT_BATCH_SIZE}"
        )
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise IdentityControlError("identity control max seconds must be positive")
    if output_path.exists() or output_path.is_symlink():
        raise IdentityControlError(f"identity output already exists: {output_path}")
    evidence_path = validation_root / "validation_slice_evidence.json"
    started_utc = track_a.utc_now()
    labels_opened = False
    try:
        assets = _load_validation_assets(
            repository_root=repository_root,
            validation_root=validation_root,
            evidence_path=evidence_path,
        )
        labels_opened = True
        track_a._configure_deterministic_execution()
        resource_bundle = track_a._load_resource_manifest(
            args.resource_manifest.resolve(),
            model_path=args.model_path,
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
        )
        preflight = track_a._resource_preflight(
            min_free_bytes=int(args.min_free_bytes),
            probe_bytes=int(args.probe_bytes),
        )
        guard = track_a.ResourceGuard(time.perf_counter(), float(args.max_seconds))
        state = track_a._load_public_state(
            model_path=args.model_path,
            model_revision=MODEL_REVISION,
            resource=resource_bundle,
            min_free_bytes=int(args.min_free_bytes),
        )
        binding = _identity_binding(
            repository_root=repository_root,
            assets=assets,
            resource=resource_bundle,
            state=state,
        )
        rows, aggregate = _forward_control(assets=assets, state=state, guard=guard)
        guard.check("before identity evidence")
        evidence = {
            "schema": IDENTITY_SCHEMA,
            "task_id": TASK_ID,
            "track": "track_a",
            "status": "COMPLETED",
            "control_role": "matched public-model arithmetic diagnostic",
            "canonical_comparison_complete": False,
            "unknown_target_recovery_claim": False,
            "public_auxiliary_labels_opened": labels_opened,
            "evaluator_truth_opened": False,
            "started_utc": started_utc,
            "ended_utc": track_a.utc_now(),
            "command": " ".join(str(value) for value in [sys.executable, *sys.argv]),
            "git_head_at_execution": track_a._git_head(),
            "validation": {
                "evidence": track_a._repo_record(assets.evidence_path, repository_root),
                "selection": {
                    "rows": 24,
                    "source_rows": "[8:32)",
                    "split": "public development",
                },
                "disjointness": {
                    "checked_before_validation_label_access": True,
                    "overlap_counts": {"panel": 0, "inverse_train": 0, "target_update_train": 0, "blind_evaluation": 0},
                },
                "observation": track_a._repo_record(assets.observation_path, repository_root),
                "labels": track_a._repo_record(assets.truth_path, repository_root),
                "records": track_a._repo_record(assets.records_path, repository_root),
                "observation_metadata": assets.observation_metadata,
                "label_metadata": assets.truth_metadata,
            },
            "public_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "resource_manifest": track_a._resource_binding(resource_bundle),
                "loaded_prefix_sha256": state.prefix_digest,
                "loaded_embedding_sha256": state.embedding_digest,
                "loaded_parameter_bytes": state.parameter_bytes,
                "public_module_dtype": str(next(state.prefix.parameters()).dtype),
                "forward_output_dtype": str(state.embedding_weight.dtype),
            },
            "method_state_binding": binding,
            "preparation": {
                "training_steps": 0,
                "adaptation_steps": 0,
                "model_load_and_state_digest_seconds": state.preparation_seconds,
                "retained_model_resource_bytes": resource_bundle.total_bytes,
                "retained_loaded_parameter_bytes": state.parameter_bytes,
            },
            "resource_preflight": preflight,
            "memory": {"preparation_peak": state.preparation_peak},
            "execution": {
                "batch_size": DEFAULT_BATCH_SIZE,
                "sequence_tokens": 40,
                "cut_depth": CUT_DEPTH,
                "public_prefix_layer_evaluations": aggregate["public_prefix_layer_evaluations"],
                "candidate_prefix_simulations": 0,
                "steady_state_forward_seconds": aggregate["forward_seconds"],
                "process_max_rss_kib": int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "arithmetic_discrepancy": aggregate,
            "records": rows,
        }
        _write_json_create(output_path, evidence)
        return {
            "status": evidence["status"],
            "output": str(output_path),
            "records": aggregate["records"],
            "relative_l2_mean": aggregate["relative_l2_mean"],
            "allclose_atol_1e-2_rtol_1e-2": aggregate["allclose_atol_1e-2_rtol_1e-2"],
            "evaluator_truth_opened": False,
        }
    except Exception as exc:
        failure = output_path.with_name(output_path.stem + ".failure.json")
        if not failure.exists() and not failure.is_symlink():
            try:
                _write_json_create(
                    failure,
                    {
                        "schema": IDENTITY_SCHEMA,
                        "task_id": TASK_ID,
                        "track": "track_a",
                        "status": "FAILED_CLOSED",
                        "control_role": "matched public-model arithmetic diagnostic",
                        "public_auxiliary_labels_opened": labels_opened,
                        "evaluator_truth_opened": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "started_utc": started_utc,
                        "ended_utc": track_a.utc_now(),
                        "git_head_at_execution": track_a._git_head(),
                    },
                )
            except Exception:
                pass
        if isinstance(exc, IdentityControlError):
            raise
        if isinstance(exc, track_a.TrackAError):
            raise IdentityControlError(str(exc)) from exc
        raise IdentityControlError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except IdentityControlError as exc:
        print(
            json.dumps(
                {"status": "FAILED_CLOSED", "error": str(exc), "evaluator_truth_opened": False},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
