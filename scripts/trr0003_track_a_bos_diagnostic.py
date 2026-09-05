#!/usr/bin/env python3
"""Diagnose embedding-forward arithmetic and the known-BOS clamp.

This is a small public-auxiliary diagnostic for the TRR-0003 Track A pilot.
It deliberately does not modify or re-run the frozen Track A runner.  On one
or two records from the already registered, disjoint public validation slice
it compares ``forward_full(token_ids)`` with
``forward_public_embeddings(public_embedding[token_ids])`` and the cached
cut-4 activation.  It then measures the 32-step inverse cycle before and
after the runner's known-BOS clamp.  The public labels are used only to make a
matched diagnostic input; evaluator-private truth and the shared panel are
never loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource as sys_resource
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from token_reconstruction.checkpoint_inverse import (
    clamp_known_bos,
    forward_public_embeddings,
    invert_public_prefix,
)
from token_reconstruction.footing import BOS_TOKEN_ID, CUT_DEPTH, HIDDEN_SIZE, MODEL_ID, MODEL_REVISION, TASK_ID

import trr0003_track_a as track_a
import trr0003_track_a_identity as identity


SCHEMA = "token-reconstruction.trr0003-track-a-bos-forward-diagnostic.v1"
VALIDATION_ROOT = Path("outputs/TRR-0003/track_b/public_validation_slice_v2")
DEFAULT_OUTPUT = Path("outputs/TRR-0003/track_a_bos_forward_diagnostic/public_bos_forward.json")
DEFAULT_RECORD_INDICES = (0, 1)
DEFAULT_ITERATIONS = 32
DEFAULT_DAMPING = 0.5
DEFAULT_MAX_SECONDS = 1200.0


class DiagnosticError(RuntimeError):
    """Raised when the public diagnostic cannot satisfy its contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise DiagnosticError(f"refusing to overwrite diagnostic artifact: {path}")
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
        raise DiagnosticError(f"refusing to overwrite diagnostic artifact: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DiagnosticError(f"{field} is non-finite")
    return result


def _relative_l2(error: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(error.float().reshape(-1))
    denominator = torch.linalg.vector_norm(reference.float().reshape(-1)).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    reference_f = reference.float()
    error = actual_f - reference_f
    return {
        "max_abs": _finite_float(error.abs().max().item(), field="max_abs"),
        "mean_abs": _finite_float(error.abs().mean().item(), field="mean_abs"),
        "rmse": _finite_float(error.square().mean().sqrt().item(), field="rmse"),
        "relative_l2": _finite_float(_relative_l2(error, reference_f), field="relative_l2"),
        "allclose_atol_1e-3_rtol_1e-3": bool(torch.allclose(actual_f, reference_f, atol=1e-3, rtol=1e-3)),
        "allclose_atol_1e-2_rtol_1e-2": bool(torch.allclose(actual_f, reference_f, atol=1e-2, rtol=1e-2)),
    }


def _select_indices(values: Sequence[int], *, record_count: int = 24) -> tuple[int, ...]:
    if len(values) not in (1, 2):
        raise DiagnosticError("diagnostic requires one or two record indices")
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise DiagnosticError("diagnostic record indices must be distinct")
    if any(value < 0 or value >= record_count for value in result):
        raise DiagnosticError("diagnostic record index is outside the public validation slice")
    return result


def _code_binding(repository_root: Path, resource: track_a.ResourceBundle) -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(track_a.__file__).resolve(),
        Path(identity.__file__).resolve(),
        repository_root / "src/token_reconstruction/checkpoint_inverse.py",
        repository_root / "src/token_reconstruction/public_prefix.py",
        repository_root / "src/token_reconstruction/footing.py",
    )
    return {
        "code": [track_a._repo_record(path, repository_root) for path in paths],
        "code_commit": track_a._git_head(),
        "resource_manifest": track_a._resource_binding(resource),
    }


def _per_position_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if actual.ndim != 2 or reference.ndim != 2 or actual.shape != reference.shape:
        raise DiagnosticError("per-position metric geometry differs")
    post = _metrics(actual[1:], reference[1:])
    bos = _metrics(actual[:1], reference[:1])
    return {"all_positions": _metrics(actual, reference), "bos_position": bos, "post_bos": post}


def _forward_record(
    *,
    input_ids: torch.Tensor,
    cached: torch.Tensor,
    state: track_a.LoadedPublicState,
    embedding_weight: torch.Tensor,
) -> dict[str, Any]:
    token_forward = state.prefix.forward_full(input_ids)
    embedding_input = embedding_weight[input_ids]
    embedding_forward = forward_public_embeddings(state.prefix, embedding_input)
    for name, value in (("token_forward", token_forward), ("embedding_forward", embedding_forward)):
        if tuple(value.shape) != tuple(cached.shape):
            raise DiagnosticError(f"{name} geometry differs from cached H4")
        if not torch.isfinite(value).all().item():
            raise DiagnosticError(f"{name} produced non-finite H4")
    return {
        "token_forward_vs_embedding_forward": _per_position_metrics(token_forward[0], embedding_forward[0]),
        "token_forward_vs_cached_h4": _per_position_metrics(token_forward[0], cached[0]),
        "embedding_forward_vs_cached_h4": _per_position_metrics(embedding_forward[0], cached[0]),
    }


def _cycle_record(
    *,
    target: torch.Tensor,
    state: track_a.LoadedPublicState,
    iterations: int,
    damping: float,
) -> dict[str, Any]:
    result = invert_public_prefix(
        state.prefix,
        target,
        iterations=iterations,
        damping=damping,
    )
    if not result.all_finite:
        raise DiagnosticError("inverse estimate or branch trace became non-finite")
    before = result.embedding_estimate
    after = clamp_known_bos(before, state.embedding_weight, bos_token_id=BOS_TOKEN_ID)
    bos_embedding = state.embedding_weight[BOS_TOKEN_ID].to(device=before.device, dtype=before.dtype)
    bos_error_before = _metrics(before[0, 0].reshape(1, -1), bos_embedding.reshape(1, -1))
    bos_error_after = _metrics(after[0, 0].reshape(1, -1), bos_embedding.reshape(1, -1))
    cycle_before = forward_public_embeddings(state.prefix, before)
    cycle_after = forward_public_embeddings(state.prefix, after)
    if not torch.isfinite(cycle_before).all().item() or not torch.isfinite(cycle_after).all().item():
        raise DiagnosticError("cycle forward became non-finite")
    target_row = target[0]
    before_metrics = _per_position_metrics(cycle_before[0], target_row)
    after_metrics = _per_position_metrics(cycle_after[0], target_row)
    branch_trace = [item.as_dict() for item in result.branch_stats_reverse_order]
    return {
        "iterations": iterations,
        "damping": damping,
        "inverse_all_finite": True,
        "bos_embedding_error_before_clamp": bos_error_before,
        "bos_embedding_error_after_clamp": bos_error_after,
        "cycle_residual_before_clamp": before_metrics,
        "cycle_residual_after_clamp": after_metrics,
        "post_bos_residual_change_after_minus_before": {
            "relative_l2": after_metrics["post_bos"]["relative_l2"] - before_metrics["post_bos"]["relative_l2"],
            "rmse": after_metrics["post_bos"]["rmse"] - before_metrics["post_bos"]["rmse"],
        },
        "branch_trace_reverse_order": branch_trace,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=VALIDATION_ROOT)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record-indices", type=int, nargs="+", default=list(DEFAULT_RECORD_INDICES))
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--min-free-bytes", type=int, default=track_a.DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--probe-bytes", type=int, default=track_a.DEFAULT_PROBE_BYTES)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    validation_root = args.validation_root.resolve()
    output_path = args.output.resolve()
    indices = _select_indices(args.record_indices)
    if args.iterations <= 0:
        raise DiagnosticError("iterations must be positive")
    if not math.isfinite(args.damping) or not 0.0 < args.damping <= 1.0:
        raise DiagnosticError("damping must lie in (0,1]")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise DiagnosticError("max_seconds must be positive")
    if output_path.exists() or output_path.is_symlink():
        raise DiagnosticError(f"diagnostic output already exists: {output_path}")
    started_utc = utc_now()
    labels_opened = False
    try:
        # This helper validates the disjointness receipt before opening its
        # public auxiliary labels.  It never reads the evaluator sidecar.
        assets = identity._load_validation_assets(
            repository_root=repository_root,
            validation_root=validation_root,
            evidence_path=validation_root / "validation_slice_evidence.json",
        )
        labels_opened = True
        track_a._configure_deterministic_execution()
        resource = track_a._load_resource_manifest(
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
            resource=resource,
            min_free_bytes=int(args.min_free_bytes),
        )
        binding = _code_binding(repository_root, resource)
        records: list[dict[str, Any]] = []
        started = time.perf_counter()
        for record_index in indices:
            guard.check(f"record {record_index} forward/cycle diagnostic")
            valid_tokens = 40
            input_ids = assets.truth[record_index : record_index + 1, :valid_tokens].to(device="cuda", dtype=torch.long)
            cached = assets.observation[record_index : record_index + 1, :valid_tokens].to(device="cuda")
            if input_ids[:, 0].ne(BOS_TOKEN_ID).any().item():
                raise DiagnosticError("public validation diagnostic input lacks BOS")
            if not torch.isfinite(cached).all().item():
                raise DiagnosticError("cached validation H4 is non-finite")
            track_a.synchronize()
            record_started = time.perf_counter()
            forward = _forward_record(
                input_ids=input_ids,
                cached=cached,
                state=state,
                embedding_weight=state.embedding_weight,
            )
            cycle = _cycle_record(
                target=cached.float(),
                state=state,
                iterations=int(args.iterations),
                damping=float(args.damping),
            )
            track_a.synchronize()
            records.append({
                "record_index": record_index,
                "record_id": assets.record_ids[record_index],
                "valid_tokens": valid_tokens,
                "forward": forward,
                "cycle": cycle,
                "inference_seconds": time.perf_counter() - record_started,
                "public_prefix_layer_evaluations": CUT_DEPTH * 5 + int(args.iterations) * 2 * CUT_DEPTH,
                "candidate_prefix_simulations": 0,
            })
            del input_ids, cached
            track_a.synchronize()
        guard.check("before diagnostic evidence")
        evidence = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "track": "track_a",
            "status": "COMPLETED",
            "diagnostic_role": "public auxiliary forward-path and known-BOS clamp contribution",
            "canonical_comparison_complete": False,
            "unknown_target_recovery_claim": False,
            "public_auxiliary_labels_opened": labels_opened,
            "evaluator_truth_opened": False,
            "shared_panel_loaded": False,
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": " ".join(str(value) for value in [sys.executable, *sys.argv]),
            "git_head_at_execution": track_a._git_head(),
            "selection": {
                "validation_root": str(validation_root),
                "record_indices": list(indices),
                "record_ids": [assets.record_ids[i] for i in indices],
                "valid_tokens": 40,
                "source_rows": [8 + i for i in indices],
                "disjointness_evidence": track_a._repo_record(assets.evidence_path, repository_root),
                "overlap_counts": {"panel": 0, "inverse_train": 0, "target_update_train": 0, "blind_evaluation": 0},
                "checked_before_validation_label_access": True,
            },
            "public_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "cut_depth": CUT_DEPTH,
                "hidden_size": HIDDEN_SIZE,
                "resource_manifest": track_a._resource_binding(resource),
                "loaded_prefix_sha256": state.prefix_digest,
                "loaded_embedding_sha256": state.embedding_digest,
                "loaded_parameter_bytes": state.parameter_bytes,
                "public_module_dtype": str(next(state.prefix.parameters()).dtype),
                "forward_output_dtype": str(state.embedding_weight.dtype),
                "inverse_accumulation_dtype": "torch.float32",
            },
            "fixed_execution": {
                "iterations": int(args.iterations),
                "damping": float(args.damping),
                "forward_token_path": "public_prefix.forward_full(token_ids)",
                "forward_embedding_path": "checkpoint_inverse.forward_public_embeddings(public_embedding[token_ids])",
                "cycle_path": "forward_public_embeddings(prefix, estimate)",
                "known_bos_clamp": "clamp_known_bos(estimate, embedding_weight, bos_token_id=128000)",
                "position_ids": "arange(valid_length)",
                "attention": "full causal mask; deterministic SDPA math",
                "candidate_prefix_simulations": 0,
                "fitted_parameters": False,
                "teacher_prefix_diagnostic": False,
            },
            "method_state_binding": binding,
            "preparation": {
                "training_steps": 0,
                "adaptation_steps": 0,
                "model_load_and_state_digest_seconds": state.preparation_seconds,
                "retained_model_resource_bytes": resource.total_bytes,
                "retained_loaded_parameter_bytes": state.parameter_bytes,
            },
            "resource_preflight": preflight,
            "memory": {
                "preparation_peak": state.preparation_peak,
                "diagnostic_peak": {
                    "process_max_rss_kib": int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                },
            },
            "records": records,
            "aggregate": {
                "records": len(records),
                "inference_seconds": time.perf_counter() - started,
                "public_prefix_layer_evaluations": sum(int(row["public_prefix_layer_evaluations"]) for row in records),
                "candidate_prefix_simulations": 0,
            },
        }
        _json_create(output_path, evidence)
        return {"status": evidence["status"], "output": str(output_path), "records": len(records), "evaluator_truth_opened": False}
    except Exception as exc:
        failure_path = output_path.with_name(output_path.stem + ".failure.json")
        if not failure_path.exists() and not failure_path.is_symlink():
            try:
                _json_create(
                    failure_path,
                    {
                        "schema": SCHEMA,
                        "task_id": TASK_ID,
                        "track": "track_a",
                        "status": "FAILED_CLOSED",
                        "error": f"{type(exc).__name__}: {exc}",
                        "started_utc": started_utc,
                        "ended_utc": utc_now(),
                        "record_indices": list(indices),
                        "public_auxiliary_labels_opened": labels_opened,
                        "evaluator_truth_opened": False,
                        "shared_panel_loaded": False,
                        "git_head_at_execution": track_a._git_head(),
                    },
                )
            except Exception:
                pass
        raise


def main() -> int:
    try:
        result = run(build_parser().parse_args())
    except (DiagnosticError, identity.IdentityControlError, track_a.TrackAError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-0003 Track A BOS/forward diagnostic failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
