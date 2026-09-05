#!/usr/bin/env python3
"""Truth-free reconstruction for one opaque TRR-P01 public arm.

The process receives one condition-free observation interface and the public
prototype table.  It never accepts a target model, source record, dataset,
condition label, truth tensor, or correctness signal.  Static lookup and the
optional fixed reference correction are committed together in one prediction
artifact before the private scorer is invoked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

from safetensors.torch import save_file
import torch

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
# The published historical lens loader lives under ``reference`` at the
# repository root.  Direct script execution places only this script directory
# on sys.path, so add the worktree root explicitly before the lazy import.
for _path in (_SOURCE_ROOT, _SOURCE_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from token_reconstruction.public_prefix import ContiguousPublicPrefix  # noqa: E402
from token_reconstruction.trr_p01 import (  # noqa: E402
    PrototypeTable,
    apply_reference_correction,
    nearest_embedding,
)
from common import (  # noqa: E402
    BOS_TOKEN_ID,
    CORRECTION_METHOD,
    CUT_DEPTH,
    EVIDENCE_SCHEMA,
    HIDDEN_SIZE,
    VOCAB_SIZE,
    METHODS,
    METRICS,
    MODEL_ID,
    MODEL_REVISION,
    PREDICTION_SCHEMA,
    REFERENCE_TOKEN,
    SCORED_TOKENS,
    SEQUENCE_TOKENS,
    TASK_ID,
    digest_tensor,
    environment_record,
    estimate_resource_need,
    file_record,
    load_json,
    load_public_interface,
    load_public_model,
    peak_memory,
    require_create_only_directory,
    require_create_only_file,
    resource_guard,
    seed_everything,
    sha256_file,
    utc_now,
    validate_contiguous_observations,
    write_json_exclusive,
    write_jsonl_exclusive,
    command_record,
)


ROUTE_SCHEMA = "token-reconstruction.trr-p01-static-route.v1"
FINISH_SCHEMA = "token-reconstruction.trr-p01-reconstruction-finish.v1"
CORRECTION_SCHEMA = "token-reconstruction.trr-p01-reference-correction.v1"
MODEL_BYTES_ESTIMATE = 2_500_000_000
# The historical fixed-K256 qualification separately exercises the copied
# cache geometry.  Include a conservative cache margin here so the final
# matrix fails closed if the live host/device no longer has room for it.
HISTORICAL_CACHE_MARGIN_BYTES = 2_000_000_000
HISTORICAL_METHODS = ("historical_a1", "historical_a1_a2_port")
PREDICTION_ARMS = (
    "boundary.cosine",
    "boundary.l2",
    "raw_embedding.cosine",
    "raw_embedding.l2",
    "reference_corrected.cosine",
    "reference_corrected.l2",
    "historical_a1.cosine",
    "historical_a1_a2_port.cosine",
)
_ALLOWED_METHODS = set((*METHODS, CORRECTION_METHOD, *HISTORICAL_METHODS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--enable-correction", action="store_true")
    parser.add_argument("--static-evidence", type=Path, default=None)
    parser.add_argument("--historical-lens", type=Path, default=None)
    parser.add_argument("--implementation-commit", default=None)
    return parser.parse_args()


def _require_public_table(config: dict[str, Any], prototype_path: Path) -> str:
    entry = config.get("prototype")
    if not isinstance(entry, dict):
        raise RuntimeError("sanitized config has no prototype identity")
    actual = sha256_file(prototype_path)
    if str(entry.get("sha256")) != actual:
        raise RuntimeError("prototype hash differs from sanitized config")
    if int(entry.get("bytes", -1)) != prototype_path.stat().st_size:
        raise RuntimeError("prototype byte count differs from sanitized config")
    return actual


def _resource_estimate(prototype_path: Path, *, historical_enabled: bool) -> dict[str, Any]:
    """Estimate the largest selected public method cell before model load."""

    normalized_embedding_bytes = VOCAB_SIZE * HIDDEN_SIZE * 4 if historical_enabled else 0
    cache_margin_bytes = HISTORICAL_CACHE_MARGIN_BYTES if historical_enabled else 0
    estimate = estimate_resource_need(
        table_bytes=prototype_path.stat().st_size,
        model_bytes=MODEL_BYTES_ESTIMATE + normalized_embedding_bytes + cache_margin_bytes,
        query_rows=256,
        prototype_chunk=8192,
    )
    return {
        **estimate,
        "records": 16,
        "sequence_tokens": SEQUENCE_TOKENS,
        "scored_tokens": SCORED_TOKENS,
        "historical_cell_enabled": historical_enabled,
        "normalized_embedding_bytes": normalized_embedding_bytes,
        "historical_cache_margin_bytes": cache_margin_bytes,
    }


def _prediction_matrix(predicted: torch.Tensor) -> torch.Tensor:
    value = predicted.detach().cpu().to(torch.int32).contiguous()
    if tuple(value.shape) != (16, SEQUENCE_TOKENS):
        raise RuntimeError("prediction matrix geometry changed")
    if value[:, 0].ne(BOS_TOKEN_ID).any().item() or value[:, 1:].lt(0).any().item() or value[:, 1:].ge(128256).any().item():
        raise RuntimeError("prediction matrix contains an invalid token")
    return value


def _static_predictions(
    observations: torch.Tensor,
    table: PrototypeTable,
    embedding_table: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    validate_contiguous_observations(observations)
    queries = observations[:, 1:, :].reshape(-1, HIDDEN_SIZE)
    result: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, torch.Tensor] = {}
    method_timings: dict[str, float] = {}
    for metric in METRICS:
        method_started = time.perf_counter()
        nearest = table.nearest(queries, metric=metric)
        value = torch.full((observations.shape[0], SEQUENCE_TOKENS), BOS_TOKEN_ID, dtype=torch.int32)
        value[:, 1:] = nearest.predictions.view(observations.shape[0], SCORED_TOKENS).to(torch.int32)
        result[f"boundary.{metric}"] = _prediction_matrix(value)
        diagnostics[f"boundary.{metric}.scores"] = nearest.scores.view(observations.shape[0], SCORED_TOKENS)
        diagnostics[f"boundary.{metric}.margins"] = nearest.margins.view(observations.shape[0], SCORED_TOKENS)
        method_timings[f"boundary.{metric}"] = time.perf_counter() - method_started
    for metric in METRICS:
        method_started = time.perf_counter()
        nearest = nearest_embedding(queries, embedding_table, metric=metric)
        value = torch.full((observations.shape[0], SEQUENCE_TOKENS), BOS_TOKEN_ID, dtype=torch.int32)
        value[:, 1:] = nearest.predictions.view(observations.shape[0], SCORED_TOKENS).to(torch.int32)
        result[f"raw_embedding.{metric}"] = _prediction_matrix(value)
        diagnostics[f"raw_embedding.{metric}.scores"] = nearest.scores.view(observations.shape[0], SCORED_TOKENS)
        diagnostics[f"raw_embedding.{metric}.margins"] = nearest.margins.view(observations.shape[0], SCORED_TOKENS)
        method_timings[f"raw_embedding.{metric}"] = time.perf_counter() - method_started
    return result, diagnostics, method_timings


def _reference_predictions(
    observations: torch.Tensor,
    prefix: ContiguousPublicPrefix,
    table: PrototypeTable,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    predictions: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, torch.Tensor] = {}
    method_timings: dict[str, float] = {}
    references = 0
    persistent = 0
    probes = 0
    started = time.perf_counter()
    for metric in METRICS:
        method_started = time.perf_counter()
        result = apply_reference_correction(
            observations=observations,
            public_prefix=prefix,
            prototypes=table,
            metric=metric,
            reference_token=REFERENCE_TOKEN,
            bos_token_id=BOS_TOKEN_ID,
            device=device,
        )
        predictions[f"{CORRECTION_METHOD}.{metric}"] = _prediction_matrix(result.predictions)
        diagnostics[f"{CORRECTION_METHOD}.{metric}.scores"] = result.scores[:, 1:]
        diagnostics[f"{CORRECTION_METHOD}.{metric}.margins"] = result.margins[:, 1:]
        # Preserve the public probe offsets as a raw diagnostic artifact.
        # They are derived only from the public reference token and the
        # reconstructed-prefix cache, and are never used to route or score a
        # later arm.
        diagnostics[f"{CORRECTION_METHOD}.{metric}.offsets"] = result.offsets
        # The fixed rule is run independently for each declared metric; counts
        # therefore make the two metric-specific public probe costs explicit.
        references += result.reference_evaluations
        persistent += result.persistent_cache_commits
        probes += result.probe_cache_commits
        method_timings[f"{CORRECTION_METHOD}.{metric}"] = time.perf_counter() - method_started
    # Each correction metric uses one scalar public-prefix call for the BOS
    # commit and one reference probe plus one reconstructed-token commit per
    # scored position.  The implementation is record-wise, so calls and
    # input-token evaluations happen to have the same count for this arm; the
    # fields remain separate for comparison with the batched historical port.
    public_prefix_calls = persistent + probes
    return predictions, diagnostics, {
        "reference_evaluations": references,
        "reference_probe_token_evaluations": references,
        "persistent_cache_commits": persistent,
        "probe_cache_commits": probes,
        "public_prefix_calls": public_prefix_calls,
        "public_prefix_input_token_evaluations": persistent + probes,
        "candidate_simulations": 0,
        "method_seconds": method_timings,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _historical_predictions(
    observations: torch.Tensor,
    prefix: ContiguousPublicPrefix,
    lens_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Run the published fixed K=256 control only when explicitly requested."""

    from token_reconstruction.trr_p01.historical_comparators import (
        HISTORICAL_LENS_ARTIFACT_SHA256,
        HISTORICAL_POLICY_ID,
        load_published_frozen_lens,
        run_fixed_k256_a1_a2,
    )

    if sha256_file(lens_path) != HISTORICAL_LENS_ARTIFACT_SHA256:
        raise RuntimeError("published historical lens hash changed")
    lens = load_published_frozen_lens(lens_path, device=device)
    result = run_fixed_k256_a1_a2(
        observations=observations,
        public_prefix=prefix,
        frozen_lens=lens,
        device=device,
        record_batch_size=8,
    )
    a1 = torch.full_like(result.predictions, BOS_TOKEN_ID, dtype=torch.int32)
    a1[:, 1:] = result.candidates[:, 1:, 0].to(torch.int32)
    predictions = {
        "historical_a1.cosine": _prediction_matrix(a1),
        "historical_a1_a2_port.cosine": _prediction_matrix(result.predictions),
    }
    diagnostics = {
        "historical_a1.cosine.scores": result.proposal_scores[:, 1:, 0],
        "historical_a1_a2_port.cosine.scores": result.selection_scores[:, 1:, :].amax(dim=2),
    }
    return predictions, diagnostics, {
        "policy_id": HISTORICAL_POLICY_ID,
        "lens_sha256": HISTORICAL_LENS_ARTIFACT_SHA256,
        "a1_forward_calls": result.a1_forward_calls,
        "a1_input_token_evaluations": result.a1_input_token_evaluations,
        "candidate_simulations": result.candidate_simulations,
        "executed_candidate_simulations": result.executed_candidate_simulations,
        "persistent_cache_commits": result.persistent_cache_commits,
        "candidate_cache_commits": result.candidate_cache_commits,
        "public_prefix_calls": result.public_prefix_calls,
        # Candidate and persistent cache writes are token instances, whereas
        # public_prefix_calls counts batched cache invocations.
        "public_prefix_input_token_evaluations": result.persistent_cache_commits + result.candidate_cache_commits,
        "method_seconds": {
            "historical_a1.cosine": result.proposal_elapsed_seconds,
            "historical_a1_a2_port.cosine": result.elapsed_seconds,
        },
        "elapsed_seconds": result.elapsed_seconds,
        "port": True,
    }


def _prediction_rows(
    predictions: dict[str, torch.Tensor],
    *,
    index: dict[str, Any],
    config_hash: str,
    evidence_hash: str,
    table_hash: str,
) -> list[dict[str, Any]]:
    ordered_ids = [str(row["record_id"]) for row in index["records"]]
    output: list[dict[str, Any]] = []
    for key, value in predictions.items():
        method, metric = key.rsplit(".", 1)
        if method not in _ALLOWED_METHODS or metric not in METRICS:
            raise RuntimeError(f"prediction key is not a frozen method/metric: {key}")
        value = _prediction_matrix(value)
        for row_index, record_id in enumerate(ordered_ids):
            tokens = [int(token) for token in value[row_index].tolist()]
            output.append(
                {
                    "record_id": record_id,
                    "method": method,
                    "metric": metric,
                    "sequence_length": SEQUENCE_TOKENS,
                    "prediction_tokens": tokens,
                    "mask_digest": index["records"][row_index]["mask_digest"],
                    "position_digest": index["records"][row_index]["position_digest"],
                    "observation_digest": index["records"][row_index]["observation_digest"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "cut_depth": CUT_DEPTH,
                    "vocab_size": 128256,
                    "hidden_size": HIDDEN_SIZE,
                    "config_sha256": config_hash,
                    "evidence_sha256": evidence_hash,
                    "table_sha256": table_hash,
                    "truth_opened": False,
                }
            )
    return output


def _write_prediction_artifacts(
    root: Path,
    predictions: dict[str, torch.Tensor],
    diagnostics: dict[str, torch.Tensor],
    *,
    index: dict[str, Any],
    config_path: Path,
    evidence_path: Path,
    table_hash: str,
) -> tuple[Path, Path, Path, dict[str, float]]:
    config_hash = sha256_file(config_path)
    evidence_hash = sha256_file(evidence_path)
    io_timing: dict[str, float] = {}
    prediction_path = root / "predictions.safetensors"
    require_create_only_file(prediction_path)
    io_started = time.perf_counter()
    save_file(
        {key: value.to(torch.int32).contiguous() for key, value in predictions.items()},
        prediction_path,
        metadata={"schema": PREDICTION_SCHEMA, "task_id": TASK_ID, "truth_opened": "false"},
    )
    io_timing["predictions_safetensors_save_seconds"] = time.perf_counter() - io_started
    rows_path = root / "predictions.jsonl"
    io_started = time.perf_counter()
    write_jsonl_exclusive(
        rows_path,
        _prediction_rows(
            predictions,
            index=index,
            config_hash=config_hash,
            evidence_hash=evidence_hash,
            table_hash=table_hash,
        ),
    )
    io_timing["predictions_jsonl_save_seconds"] = time.perf_counter() - io_started
    scores_path = root / "lookup_diagnostics.safetensors"
    require_create_only_file(scores_path)
    io_started = time.perf_counter()
    save_file(
        {key: value.float().contiguous() for key, value in diagnostics.items()},
        scores_path,
        metadata={
            "schema": "token-reconstruction.trr-p01-lookup-diagnostics.v1",
            "task_id": TASK_ID,
            "truth_opened": "false",
        },
    )
    io_timing["lookup_diagnostics_save_seconds"] = time.perf_counter() - io_started
    return prediction_path, rows_path, scores_path, io_timing


def main() -> int:
    args = parse_args()
    forbidden_options = {
        "--truth",
        "--source",
        "--dataset",
        "--target-model",
        "--target-lora",
        "--condition",
    }
    if any(value in forbidden_options for value in sys.argv[1:]):
        raise RuntimeError("reconstruction accepts only the public opaque interface")
    config, index, observations, config_path, index_path, observation_path = load_public_interface(
        args.input_root
    )
    # A prepared arm may optionally omit the explicit matrix for compatibility
    # with the earlier static diagnostic.  Once an evaluator declares the
    # final matrix, require every declared arm and every corresponding flag
    # before loading the model, so a partial prediction set cannot reach a
    # truth-scoring step.
    declared_arms = config.get("prediction_arms")
    if declared_arms is not None:
        if not isinstance(declared_arms, list) or tuple(declared_arms) != PREDICTION_ARMS:
            raise RuntimeError("sanitized prediction arm declaration changed")
        # The static arm order is boundary cosine/L2 followed by raw embedding
        # cosine/L2, matching the frozen METRICS order.
        requested_keys = [
            "boundary.cosine",
            "boundary.l2",
            "raw_embedding.cosine",
            "raw_embedding.l2",
        ]
        if args.enable_correction:
            requested_keys.extend(("reference_corrected.cosine", "reference_corrected.l2"))
        if args.historical_lens is not None:
            requested_keys.extend(("historical_a1.cosine", "historical_a1_a2_port.cosine"))
        if tuple(requested_keys) != PREDICTION_ARMS:
            raise RuntimeError(
                "declared prediction arms require --enable-correction and "
                "--historical-lens before reconstruction"
            )
    prototype_path = args.prototype.resolve()
    prototype_hash = _require_public_table(config, prototype_path)
    if args.static_evidence is not None:
        evidence = load_json(args.static_evidence.resolve())
        if evidence.get("truth_opened") is not False or evidence.get("task_id") != TASK_ID:
            raise RuntimeError("static evidence is not truth-free")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    root = require_create_only_directory(args.output_root.resolve())
    started_utc = utc_now()
    estimate = _resource_estimate(
        prototype_path, historical_enabled=args.historical_lens is not None
    )
    pre_model_guard = resource_guard(
        device=device,
        required_bytes=int(estimate["guard_required_bytes"]),
        allocation_bytes=0,
    )
    preflight_path = root / "preflight.json"
    write_json_exclusive(
        preflight_path,
        {
            "schema": "token-reconstruction.trr-p01-reconstruction-preflight.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "source_truth_included": False,
            "selected_device": str(device),
            "started_utc": started_utc,
            "implementation_commit": args.implementation_commit or "UNBOUND_PRECOMMIT",
            "input": {
                "config": file_record(config_path),
                "observation_index": file_record(index_path),
                "observations": file_record(observation_path),
            },
            "prototype": {
                "path": str(prototype_path),
                "bytes": prototype_path.stat().st_size,
                "sha256": prototype_hash,
            },
            "estimate": estimate,
            "guard": pre_model_guard,
            "numerics": {
                "deterministic_algorithms": True,
                "float32_distance": True,
                "tf32": False,
            },
        },
    )
    seed_everything(1701, device)
    phases: list[dict[str, Any]] = []

    phase_started = time.perf_counter()
    table = PrototypeTable.load(
        prototype_path,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_cut_depth=CUT_DEPTH,
        expected_vocab_size=VOCAB_SIZE,
        expected_hidden_size=HIDDEN_SIZE,
    )
    post_table_guard = resource_guard(
        device=device,
        required_bytes=int(estimate["guard_required_bytes"]),
        allocation_bytes=0,
    )
    phases.append({"phase": "load_public_prototype_table", "elapsed_seconds": time.perf_counter() - phase_started})

    phase_started = time.perf_counter()
    model = load_public_model(device=device, model_path=args.model_path)
    prefix = ContiguousPublicPrefix(model, CUT_DEPTH).to(device).eval()
    embedding_table = prefix.embed_tokens.weight.detach().cpu().contiguous()
    post_model_guard = resource_guard(
        device=device,
        required_bytes=int(estimate["guard_required_bytes"]),
        allocation_bytes=0,
    )
    phases.append({"phase": "load_public_model_and_prefix", "elapsed_seconds": time.perf_counter() - phase_started})

    phase_started = time.perf_counter()
    predictions, diagnostics, method_timings = _static_predictions(
        observations, table, embedding_table
    )
    phases.append({"phase": "static_full_vocabulary_lookup", "elapsed_seconds": time.perf_counter() - phase_started})

    optional_method_guard: dict[str, Any] | None = None
    if args.enable_correction or args.historical_lens is not None:
        # This check is repeated immediately before the expensive causal arms;
        # the largest-cell estimate already includes the fixed K256 cache
        # margin when the historical arm is selected.
        optional_method_guard = resource_guard(
            device=device,
            required_bytes=int(estimate["guard_required_bytes"]),
            allocation_bytes=0,
        )

    correction_info: dict[str, Any] | None = None
    if args.enable_correction:
        correction_predictions, correction_diagnostics, correction_info = _reference_predictions(
            observations, prefix, table, device=device
        )
        predictions.update(correction_predictions)
        diagnostics.update(correction_diagnostics)
        method_timings.update(correction_info.get("method_seconds", {}))
        phases.append({"phase": "reference_220_correction", "elapsed_seconds": correction_info["elapsed_seconds"]})

    historical_info: dict[str, Any] | None = None
    if args.historical_lens is not None:
        historical_predictions, historical_diagnostics, historical_info = _historical_predictions(
            observations, prefix, args.historical_lens.resolve(), device=device
        )
        predictions.update(historical_predictions)
        diagnostics.update(historical_diagnostics)
        method_timings.update(historical_info.get("method_seconds", {}))
        phases.append({"phase": "historical_fixed_k256_geometry_port", "elapsed_seconds": historical_info["elapsed_seconds"]})

    post_method_guard = resource_guard(
        device=device,
        required_bytes=int(estimate["guard_required_bytes"]),
        allocation_bytes=0,
    )

    source_files = [
        file_record(_SOURCE_ROOT / "src/token_reconstruction/trr_p01/boundary_prototype.py"),
        file_record(_SOURCE_ROOT / "scripts/trr_p01/common.py"),
        file_record(Path(__file__)),
    ]
    if args.historical_lens is not None:
        source_files.append(file_record(_SOURCE_ROOT / "src/token_reconstruction/trr_p01/historical_comparators.py"))
    evidence_path = root / "reconstructor_evidence.json"
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": False,
        "status": "PREDICTIONS_FROZEN_BEFORE_TRUTH",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "command": command_record(device),
        "environment": environment_record(device),
        "implementation_commit": args.implementation_commit or "UNBOUND_PRECOMMIT",
        "selected_device": str(device),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "cut_depth": CUT_DEPTH},
        "input": {
            "config": file_record(config_path),
            "observation_index": file_record(index_path),
            "observations": file_record(observation_path),
        },
        "prototype": file_record(prototype_path),
        "preflight": file_record(preflight_path),
        "methods": list(predictions),
        "method_timings": method_timings,
        "records": int(observations.shape[0]),
        "scored_tokens": int(observations.shape[0]) * SCORED_TOKENS,
        "candidate_simulations": int((correction_info or {}).get("candidate_simulations", 0)) + int((historical_info or {}).get("candidate_simulations", 0)),
        # Keep invocation counts separate from the number of token instances
        # processed by those invocations.  The historical fixed-K256 port
        # batches candidate rows, so its public-prefix call count is much
        # smaller than its token evaluation count.
        "public_prefix_calls": int((correction_info or {}).get("public_prefix_calls", 0)) + int((historical_info or {}).get("public_prefix_calls", 0)),
        "public_prefix_input_token_evaluations": int((correction_info or {}).get("public_prefix_input_token_evaluations", 0)) + int((historical_info or {}).get("public_prefix_input_token_evaluations", 0)),
        "historical_a1_forward_calls": int((historical_info or {}).get("a1_forward_calls", 0)),
        "historical_a1_input_token_evaluations": int((historical_info or {}).get("a1_input_token_evaluations", 0)),
        "phases": phases,
        "resource": {
            "estimate": estimate,
            "pre_model_guard": pre_model_guard,
            "post_table_guard": post_table_guard,
            "post_model_guard": post_model_guard,
            "pre_optional_method_guard": optional_method_guard,
            "post_method_guard": post_method_guard,
        },
        "peak_memory": peak_memory(device),
        "reference_correction": correction_info,
        "historical_control": historical_info,
        "historical_lens": file_record(args.historical_lens.resolve()) if args.historical_lens is not None else None,
        "static_evidence": file_record(args.static_evidence.resolve()) if args.static_evidence is not None else None,
        "code_files": source_files,
    }
    write_json_exclusive(evidence_path, evidence)

    output_started = time.perf_counter()
    prediction_path, rows_path, scores_path, save_timing = _write_prediction_artifacts(
        root,
        predictions,
        diagnostics,
        index=index,
        config_path=config_path,
        evidence_path=evidence_path,
        table_hash=prototype_hash,
    )
    route_path = root / "route.json"
    route_started = time.perf_counter()
    write_json_exclusive(
        route_path,
        {
            "schema": ROUTE_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": False,
            "record_order": config["record_order"],
            "methods": list(predictions),
            "stopping": "all 39 post-BOS positions",
            "candidate_simulations": evidence["candidate_simulations"],
            "public_prefix_calls": evidence["public_prefix_calls"],
            "public_prefix_input_token_evaluations": evidence["public_prefix_input_token_evaluations"],
            "prediction": file_record(prediction_path),
            "prediction_rows": file_record(rows_path),
            "lookup_diagnostics": file_record(scores_path),
            "evidence": file_record(evidence_path),
            "finish_receipt": "finish_receipt.json",
        },
    )
    route_save_seconds = time.perf_counter() - route_started
    hash_started = time.perf_counter()
    artifact_records = {
        "preflight": file_record(preflight_path),
        "evidence": file_record(evidence_path),
        "predictions": file_record(prediction_path),
        "prediction_rows": file_record(rows_path),
        "lookup_diagnostics": file_record(scores_path),
        "route": file_record(route_path),
    }
    hash_seconds = time.perf_counter() - hash_started
    finish_path = root / "finish_receipt.json"
    write_json_exclusive(
        finish_path,
        {
            "schema": FINISH_SCHEMA,
            "task_id": TASK_ID,
            "status": "OUTPUTS_HASHED_AFTER_PRETRUTH_FREEZE",
            "created_utc": utc_now(),
            "truth_opened": False,
            "implementation_commit": args.implementation_commit or "UNBOUND_PRECOMMIT",
            "selected_device": str(device),
            "methods": list(predictions),
            "io": {
                **save_timing,
                "route_save_seconds": route_save_seconds,
                "artifact_hash_seconds": hash_seconds,
                "total_output_io_and_hash_seconds": time.perf_counter() - output_started,
            },
            "artifacts": artifact_records,
            "evidence_sha256": artifact_records["evidence"]["sha256"],
            "prediction_sha256": artifact_records["predictions"]["sha256"],
            "prediction_rows_sha256": artifact_records["prediction_rows"]["sha256"],
            "lookup_diagnostics_sha256": artifact_records["lookup_diagnostics"]["sha256"],
            "truth_opened_before_finish": False,
        },
    )
    print(
        {
            "status": "PREDICTIONS_FROZEN_BEFORE_TRUTH",
            "output_root": str(root),
            "finish_receipt": str(finish_path),
            "methods": list(predictions),
            "truth_opened": False,
        }
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
