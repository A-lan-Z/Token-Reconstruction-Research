#!/usr/bin/env python3
"""Run public-only post hoc bias, scale, and normalization diagnostics.

The inputs are the frozen TRR-0003 public fitting/validation assets and the
three already-fitted checkpoints.  This command performs no fitting and does
not access a target model, target truth, candidate list, or A2 fallback.  Its
metrics are development diagnostics, not an independent confirmation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource as sys_resource
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from token_reconstruction.decoder_diagnostics import (
    DEFAULT_VARIANTS,
    FREQUENCY_BINS,
    DiagnosticVariant,
    diagnose_model,
    expected_scale_invariance,
    flatten_public_labels,
    flatten_public_records,
)
from token_reconstruction.inverse import load_inverse
from token_reconstruction.standalone_decoder import (
    ResidualMLPTokenDecoder,
    TiedAffineTokenDecoder,
    load_token_decoder,
    tensor_sha256,
    validate_embedding_table,
)


HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
METHOD_IDS = (
    "angular_inverse_control",
    TiedAffineTokenDecoder.method_id,
    ResidualMLPTokenDecoder.method_id,
)
METHOD_SPECS = {
    "angular_inverse_control": {"kind": "angular"},
    TiedAffineTokenDecoder.method_id: {"kind": "tied_affine"},
    ResidualMLPTokenDecoder.method_id: {"kind": "residual_mlp", "bottleneck_size": 256},
}


class DiagnosticRunnerError(RuntimeError):
    """Raised when a public diagnostic input or output contract is invalid."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-observations", type=Path, required=True)
    parser.add_argument("--validation-truth", type=Path, required=True)
    parser.add_argument("--validation-records", type=Path, required=True)
    parser.add_argument("--fit-truth", type=Path, required=True)
    parser.add_argument("--fit-records", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--angular-state", type=Path, required=True)
    parser.add_argument("--tied-state", type=Path, required=True)
    parser.add_argument("--mlp-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise DiagnosticRunnerError("CUDA was requested but is unavailable")
    return torch.device(raw)


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise DiagnosticRunnerError(f"{label} must be a regular file: {path}")
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    path = _regular_file(path, label=role)
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_tensor(path: Path, key: str, *, role: str) -> torch.Tensor:
    path = _regular_file(path, label=role)
    state = load_file(str(path), device="cpu")
    if set(state) != {key}:
        raise DiagnosticRunnerError(f"{role} must contain exactly tensor {key!r}")
    value = state[key].contiguous()
    if not torch.isfinite(value.float()).all().item():
        raise DiagnosticRunnerError(f"{role} contains non-finite values")
    return value


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    path = _regular_file(path, label=role)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticRunnerError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DiagnosticRunnerError(f"{role} must contain a JSON object")
    return value


def _record_ids(value: Mapping[str, Any], *, role: str) -> list[str]:
    rows = value.get("records")
    if not isinstance(rows, list) or not rows:
        raise DiagnosticRunnerError(f"{role} has no records list")
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("record_id"), str):
            raise DiagnosticRunnerError(f"{role} contains a malformed record ID")
        ids.append(row["record_id"])
    if len(set(ids)) != len(ids):
        raise DiagnosticRunnerError(f"{role} contains duplicate record IDs")
    return ids


def _check_public_split(
    fit_records_path: Path,
    validation_records_path: Path,
) -> dict[str, Any]:
    """Check record disjointness before loading either public label tensor."""

    fit_meta = _load_json(fit_records_path, role="public fit records")
    validation_meta = _load_json(validation_records_path, role="public validation records")
    fit_ids = _record_ids(fit_meta, role="public fit records")
    validation_ids = _record_ids(validation_meta, role="public validation records")
    overlap = sorted(set(fit_ids).intersection(validation_ids))
    if overlap:
        raise DiagnosticRunnerError(f"public fit and validation records overlap: {overlap[:3]}")
    if validation_meta.get("disjointness_checked_before_label_access") is not True:
        raise DiagnosticRunnerError("validation record disjointness was not checked before label access")
    return {
        "fit_record_count": len(fit_ids),
        "validation_record_count": len(validation_ids),
        "fit_validation_overlap_count": len(overlap),
        "fit_records": _file_record(fit_records_path, role="public fit record manifest"),
        "validation_records": _file_record(
            validation_records_path, role="public validation record manifest"
        ),
        "validation_disjointness_metadata": {
            "disjointness_checked_before_label_access": True,
            "overlap_counts": validation_meta.get("overlap_counts", {}),
            "truth_role": "public auxiliary validation only",
        },
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_record(path: Path, *, role: str) -> dict[str, Any]:
    return _file_record(path, role=role)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory(device: torch.device) -> dict[str, int | None]:
    # Linux reports ru_maxrss in KiB; this runner is qualified on Linux.
    host_max_rss_bytes = int(sys_resource.getrusage(sys_resource.RUSAGE_SELF).ru_maxrss * 1024)
    result: dict[str, int | None] = {
        "host_max_rss_bytes": host_max_rss_bytes,
        "allocated_bytes": None,
        "reserved_bytes": None,
        "max_allocated_bytes": None,
        "max_reserved_bytes": None,
    }
    if device.type == "cuda":
        result.update(
            {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return result


def _load_method(
    method_id: str,
    path: Path,
    *,
    device: torch.device,
) -> torch.nn.Module:
    spec = METHOD_SPECS[method_id]
    if spec["kind"] == "angular":
        return load_inverse(path, hidden_size=HIDDEN_SIZE, device=device)
    return load_token_decoder(
        path,
        method_id=method_id,
        hidden_size=HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
        device=device,
        logit_scale=16.0,
        bottleneck_size=int(spec.get("bottleneck_size", 256)),
    )


def _method_state_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "angular_inverse_control": args.angular_state,
        TiedAffineTokenDecoder.method_id: args.tied_state,
        ResidualMLPTokenDecoder.method_id: args.mlp_state,
    }


def run(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise DiagnosticRunnerError("batch size must be positive")
    device = _device(args.device)
    started_at = _utc_now()
    started_clock = time.perf_counter()
    git_commit_start = _git_commit()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise DiagnosticRunnerError(f"output is create-only and already exists: {output}")

    public_input_start = time.perf_counter()
    split_evidence = _check_public_split(args.fit_records, args.validation_records)
    validation_observations = _load_tensor(
        args.validation_observations,
        "activations",
        role="public validation observations",
    )
    validation_truth = _load_tensor(
        args.validation_truth,
        "token_ids",
        role="public validation truth",
    )
    fit_truth = _load_tensor(args.fit_truth, "token_ids", role="public fit truth")
    embedding_table = _load_tensor(
        args.embedding_table,
        "embeddings",
        role="fixed public normalized embedding table",
    ).float().contiguous()
    if tuple(validation_observations.shape) != (split_evidence["validation_record_count"], 40, HIDDEN_SIZE):
        raise DiagnosticRunnerError("public validation observation geometry does not match its record manifest")
    if tuple(validation_truth.shape) != tuple(validation_observations.shape[:2]):
        raise DiagnosticRunnerError("public validation truth geometry does not match observations")
    if tuple(fit_truth.shape) != (split_evidence["fit_record_count"], 40):
        raise DiagnosticRunnerError("public fit truth geometry does not match its record manifest")
    if tuple(embedding_table.shape) != (VOCAB_SIZE, HIDDEN_SIZE):
        raise DiagnosticRunnerError("public embedding geometry changed")
    validate_embedding_table(embedding_table, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)
    public_input_load_seconds = time.perf_counter() - public_input_start
    validation_x, validation_y, record_shape = flatten_public_records(
        validation_observations,
        validation_truth,
        bos_token_id=BOS_TOKEN_ID,
    )
    fit_y = flatten_public_labels(fit_truth, bos_token_id=BOS_TOKEN_ID)
    if validation_y.numel() == 0 or fit_y.numel() == 0:
        raise DiagnosticRunnerError("public diagnostic labels are empty")
    if validation_y.lt(0).any().item() or validation_y.ge(VOCAB_SIZE).any().item():
        raise DiagnosticRunnerError("public validation token ID is out of range")
    if fit_y.lt(0).any().item() or fit_y.ge(VOCAB_SIZE).any().item():
        raise DiagnosticRunnerError("public fit token ID is out of range")

    embedding_transfer_start = time.perf_counter()
    embedding_runtime = embedding_table.to(device=device, dtype=torch.float32)
    _synchronize(device)
    embedding_transfer_seconds = time.perf_counter() - embedding_transfer_start
    embedding_memory = _peak_memory(device)
    states = _method_state_args(args)
    method_results: dict[str, Any] = {}
    for method_id in METHOD_IDS:
        state_path = _regular_file(states[method_id], label=f"{method_id} checkpoint")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        load_start = time.perf_counter()
        model = _load_method(method_id, state_path, device=device)
        _synchronize(device)
        load_seconds = time.perf_counter() - load_start
        eval_start = time.perf_counter()
        result = diagnose_model(
            model,
            method_id=method_id,
            validation_observations=validation_x,
            validation_labels=validation_y,
            fit_labels=fit_y,
            embedding_table=embedding_runtime,
            record_shape=record_shape,
            batch_size=args.batch_size,
            variants=DEFAULT_VARIANTS,
        )
        _synchronize(device)
        eval_seconds = time.perf_counter() - eval_start
        result["checkpoint"] = _file_record(state_path, role=f"{method_id} fitted checkpoint")
        result["timing"] = {
            "checkpoint_load_and_device_seconds": load_seconds,
            "diagnostic_projection_seconds": eval_seconds,
            "embedding_transfer_seconds_once_before_methods": embedding_transfer_seconds,
            "timing_scope": "post hoc diagnostic total across five variant projections; not deployed steady-state inference",
        }
        result["peak_memory"] = _peak_memory(device)
        # This is a post hoc diagnostic.  Retaining a model object beyond its
        # method would make the memory measurement misleading and is not part
        # of the deployed decoder contract.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        method_results[method_id] = result

    no_bias = "vocab_bias_disabled"
    no_bias_scale_1 = "no_bias_scale_1"
    for method_id, result in method_results.items():
        variants = result["variants"]
        scale_control = expected_scale_invariance(
            variants[no_bias], variants[no_bias_scale_1]
        )
        left, right = sorted((no_bias, no_bias_scale_1))
        pair_key = f"{left}__vs__{right}"
        empirical = result["pairwise_variant_comparisons"].get(pair_key)
        if empirical is None:
            raise DiagnosticRunnerError(f"missing empirical scale comparison for {method_id}")
        scale_control["empirical"] = empirical
        scale_control["empirical_argmax_invariant"] = (
            empirical["prediction_changed_examples"] == 0
        )
        result["scale_control"] = scale_control

    finished_at = _utc_now()
    finished_clock = time.perf_counter()
    git_commit_end = _git_commit()
    evidence = {
        "schema": "token-reconstruction.trr0004-public-decoder-diagnostic.v1",
        "task_id": "TRR-0004",
        "track": "track_b",
        "status": "posthoc_public_auxiliary_diagnostic",
        "purpose": "Diagnose whether fitted CE vocabulary bias, scale, or output normalization contributes to the standalone decoder gap.",
        "access_contract": {
            "public_fit_labels_used": True,
            "public_validation_labels_used": True,
            "target_weights_accessed": False,
            "target_truth_accessed": False,
            "a2_fallback": False,
            "candidate_simulations": 0,
            "public_prefix_calls": 0,
            "panel_truth_accessed": False,
            "current_evaluation_truth_accessed": False,
            "validation_truth_role": "public auxiliary validation only",
        },
        "warning": "Post hoc ablations of fitted checkpoints are diagnosis only; zeroing a fitted bias is not a no-bias fit and this validation is not independent confirmation.",
        "execution": {
            "command": [sys.executable, *sys.argv],
            "working_directory": str(Path.cwd().resolve()),
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": finished_clock - started_clock,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
        },
        "code": {
            "git_commit_start": git_commit_start,
            "git_commit_end": git_commit_end,
            "git_commit_unchanged": git_commit_start == git_commit_end,
            "runner": _source_record(Path(__file__), role="diagnostic runner source"),
            "diagnostic_module": _source_record(
                Path(__file__).resolve().parents[1] / "src/token_reconstruction/decoder_diagnostics.py",
                role="diagnostic module source",
            ),
            "standalone_decoder_module": _source_record(
                Path(__file__).resolve().parents[1] / "src/token_reconstruction/standalone_decoder.py",
                role="frozen standalone decoder source",
            ),
            "inverse_module": _source_record(
                Path(__file__).resolve().parents[1] / "src/token_reconstruction/inverse.py",
                role="frozen angular inverse source",
            ),
        },
        "configuration": {
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "bos_token_id": BOS_TOKEN_ID,
            "batch_size": args.batch_size,
            "device": str(device),
            "variants": [asdict(variant) for variant in DEFAULT_VARIANTS],
            "frequency_bins": [
                {"name": name, "lower_fit_count": lower, "upper_fit_count": upper}
                for name, lower, upper in FREQUENCY_BINS
            ],
            "rank_definition": "one plus the number of strictly larger full-vocabulary logits; ties share competition rank",
        },
        "public_split": split_evidence,
        "phase_timing_seconds": {
            "public_input_and_record_manifest_load": public_input_load_seconds,
            "embedding_transfer_once": embedding_transfer_seconds,
            "checkpoint_load_and_device_by_method": {
                method_id: result["timing"]["checkpoint_load_and_device_seconds"]
                for method_id, result in method_results.items()
            },
            "diagnostic_projection_by_method": {
                method_id: result["timing"]["diagnostic_projection_seconds"]
                for method_id, result in method_results.items()
            },
        },
        "memory_at_shared_embedding_boundary": embedding_memory,
        "inputs": {
            "validation_observations": _file_record(
                args.validation_observations, role="public validation observations"
            ),
            "validation_truth": _file_record(args.validation_truth, role="public validation truth"),
            "fit_truth": _file_record(args.fit_truth, role="public fit truth"),
            "embedding_table": _file_record(
                args.embedding_table, role="fixed public normalized embedding table"
            ),
            "tensor_hashes": {
                "validation_observations": tensor_sha256(validation_observations),
                "validation_truth": tensor_sha256(validation_truth),
                "fit_truth": tensor_sha256(fit_truth),
                "embedding_table": tensor_sha256(embedding_table),
            },
            "geometry": {
                "validation_observations": list(validation_observations.shape),
                "validation_truth": list(validation_truth.shape),
                "fit_truth": list(fit_truth.shape),
                "embedding_table": list(embedding_table.shape),
                "validation_flat_examples": int(validation_y.numel()),
                "fit_flat_examples": int(fit_y.numel()),
            },
        },
        "methods": method_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "diagnostic_complete", "output": str(output), "methods": list(method_results)}))
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return run(args)
    except (DiagnosticRunnerError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
