#!/usr/bin/env python3
"""Bounded CPU qualification for the largest TRR-P01 method cell.

This command uses only the public model, the public prototype table, the
published frozen A1 lens, and deterministic public probe-token inputs.  It
qualifies one representative K=256 causal simulation at position 39 for an
8-record batch (2,048 copied-cache candidate rows), one reference-token probe,
and a full-vocabulary nearest lookup.  It never accepts or opens target-model,
source-record, condition, or truth inputs.

The output is diagnostic preparation evidence.  It is not a reconstruction
score and does not replace the later opaque panel run.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from token_reconstruction.public_prefix import ContiguousPublicPrefix  # noqa: E402
from token_reconstruction.trr_p01 import PrototypeTable  # noqa: E402
from token_reconstruction.trr_p01.historical_comparators import (  # noqa: E402
    A1_TOP_K,
    A2_BUDGET,
    BOS_TOKEN_ID,
    CUT_DEPTH,
    HIDDEN_SIZE,
    HISTORICAL_LENS_ARTIFACT_SHA256,
    HISTORICAL_POLICY_ID,
    PUBLIC_MODEL_ID,
    PUBLIC_MODEL_REVISION,
    VOCAB_SIZE,
    load_published_frozen_lens,
)
from common import (  # noqa: E402
    TASK_ID,
    digest_tensor,
    environment_record,
    estimate_resource_need,
    file_record,
    load_json,
    load_public_model,
    peak_memory,
    require_create_only_directory,
    require_create_only_file,
    resource_guard,
    seed_everything,
    sha256_file,
    utc_now,
    validate_public_plan,
    write_json_exclusive,
)


QUALIFICATION_SCHEMA = "token-reconstruction.trr-p01-method-qualification.v1"
OUTPUT_SCHEMA = "token-reconstruction.trr-p01-method-qualification-output.v1"
REFERENCE_TOKEN = 220
RECORDS = 8
SEQUENCE_TOKENS = 40
PREFIX_LENGTH = SEQUENCE_TOKENS - 1
QUALIFICATION_POSITION = SEQUENCE_TOKENS - 1
RECORD_BATCH_SIZE = 8
MODEL_BYTES_ESTIMATE = 2_500_000_000


class MethodQualificationError(RuntimeError):
    """Raised when the bounded CPU qualification cannot preserve its contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--historical-lens", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--implementation-commit", default=None)
    return parser


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise MethodQualificationError(f"{label} must be a regular file: {path}")
    return path


def _finite(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all().item():
        raise MethodQualificationError(f"{label} contains non-finite values")


def _tensor_record(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().cpu().contiguous()
    _finite(value.float(), "qualification tensor")
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype).replace("torch.", ""),
        "sha256": digest_tensor(value),
    }


def _load_probe_ids(plan: Mapping[str, Any]) -> list[int]:
    qualification = plan.get("qualification")
    if not isinstance(qualification, Mapping):
        raise MethodQualificationError("frozen qualification section is missing")
    values = qualification.get("probe_token_ids")
    if not isinstance(values, list) or len(values) != 256:
        raise MethodQualificationError("frozen public probe-token geometry changed")
    try:
        ids = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise MethodQualificationError("public probe-token IDs are invalid") from exc
    if sorted(ids) != ids or len(set(ids)) != len(ids) or ids[0] < 0 or ids[-1] >= VOCAB_SIZE:
        raise MethodQualificationError("public probe-token IDs are not the fixed ascending sample")
    return ids


def _public_probe_sequences(probe_ids: Sequence[int]) -> torch.Tensor:
    # Reuse only the declared public qualification IDs.  The cyclic indexing
    # supplies 8*39 non-BOS inputs without consulting any panel or truth data.
    sequences = torch.full((RECORDS, SEQUENCE_TOKENS), BOS_TOKEN_ID, dtype=torch.long)
    for row in range(RECORDS):
        for position in range(1, SEQUENCE_TOKENS):
            sequences[row, position] = int(probe_ids[(row * PREFIX_LENGTH + position - 1) % len(probe_ids)])
    if sequences[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise MethodQualificationError("public probe BOS construction changed")
    return sequences


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise MethodQualificationError("public module has no parameters") from exc


def _repeat_cache(cache: Any, repeats: int) -> Any:
    try:
        repeated = copy.deepcopy(cache)
    except Exception as exc:  # pragma: no cover - backend-specific failure
        raise MethodQualificationError("public prefix cache cannot be copied") from exc
    repeat = getattr(repeated, "batch_repeat_interleave", None)
    if not callable(repeat):
        raise MethodQualificationError("public prefix cache cannot repeat candidate rows")
    repeat(int(repeats))
    return repeated


def _check_output(value: Any, shape: tuple[int, ...], dtype: torch.dtype, label: str) -> torch.Tensor:
    if isinstance(value, tuple):
        if not value:
            raise MethodQualificationError(f"{label} returned an empty tuple")
        value = value[0]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise MethodQualificationError(f"{label} geometry changed")
    if value.dtype != dtype:
        raise MethodQualificationError(f"{label} dtype changed: {value.dtype}")
    _finite(value.float(), label)
    return value


def _source_files(lens_path: Path) -> list[dict[str, Any]]:
    paths = (
        _SOURCE_ROOT / "scripts/trr_p01/qualify_methods.py",
        _SOURCE_ROOT / "scripts/trr_p01/common.py",
        _SOURCE_ROOT / "src/token_reconstruction/public_prefix.py",
        _SOURCE_ROOT / "src/token_reconstruction/trr_p01/boundary_prototype.py",
        _SOURCE_ROOT / "src/token_reconstruction/trr_p01/historical_comparators.py",
        _SOURCE_ROOT / "reference/strict_bos/round001_teacher.py",
        _SOURCE_ROOT / "src/token_reconstruction/a1a2_configuration_search.py",
        lens_path,
    )
    return [file_record(_regular_file(path, "source artifact")) for path in paths]


def _guard_estimate(table_path: Path) -> dict[str, Any]:
    table_bytes = int(table_path.stat().st_size)
    normalized_embedding_bytes = VOCAB_SIZE * HIDDEN_SIZE * 4
    candidate_output_bytes = RECORDS * A2_BUDGET * HIDDEN_SIZE * 2
    prefix_output_bytes = RECORDS * SEQUENCE_TOKENS * HIDDEN_SIZE * 2
    # Include the float32 normalized public embedding table and the largest
    # copied-cache output in the model-side estimate.  The common estimator
    # adds float32 full-vocab lookup workspace and a 20% margin.
    extra_bytes = normalized_embedding_bytes + candidate_output_bytes + prefix_output_bytes
    estimate = estimate_resource_need(
        table_bytes=table_bytes,
        model_bytes=MODEL_BYTES_ESTIMATE + extra_bytes,
        query_rows=RECORDS * A2_BUDGET,
        prototype_chunk=8192,
    )
    return {
        **estimate,
        "records": RECORDS,
        "sequence_tokens": SEQUENCE_TOKENS,
        "prefix_length": PREFIX_LENGTH,
        "qualification_position": QUALIFICATION_POSITION,
        "record_batch_size": RECORD_BATCH_SIZE,
        "candidate_budget": A2_BUDGET,
        "candidate_batch_rows": RECORDS * A2_BUDGET,
        "normalized_embedding_bytes": normalized_embedding_bytes,
        "candidate_output_bytes": candidate_output_bytes,
        "prefix_output_bytes": prefix_output_bytes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    forbidden = {
        "--truth",
        "--source",
        "--dataset",
        "--condition",
        "--target-model",
        "--target-lora",
    }
    if any(value in forbidden for value in sys.argv[1:]):
        raise MethodQualificationError("method qualification accepts public probes only")
    implementation_commit = args.implementation_commit or os.environ.get(
        "TRR_P01_IMPLEMENTATION_COMMIT", ""
    )
    if not implementation_commit:
        raise MethodQualificationError("an implementation commit identity is required")
    if args.model_path is not None and args.model_path.is_symlink():
        raise MethodQualificationError("model path must be a regular local directory")

    plan_path = _regular_file(args.plan.resolve(), "frozen pilot plan")
    plan = load_json(plan_path)
    validate_public_plan(plan)
    probe_ids = _load_probe_ids(plan)
    input_ids = _public_probe_sequences(probe_ids)
    lens_path = _regular_file(args.historical_lens.resolve(), "historical lens")
    lens_hash = sha256_file(lens_path)
    if lens_hash != HISTORICAL_LENS_ARTIFACT_SHA256:
        raise MethodQualificationError("published historical lens hash changed")

    build_root = args.build_root.resolve()
    prototype_path = _regular_file(
        (args.prototype or (build_root / "boundary_prototypes.safetensors")).resolve(),
        "public prototype table",
    )
    output_root = require_create_only_directory(args.output_root.resolve())
    started_utc = utc_now()
    started = time.perf_counter()
    device = torch.device("cpu")
    seed_everything(1701, device)
    estimate = _guard_estimate(prototype_path)
    pre_guard = resource_guard(
        device=device,
        required_bytes=int(estimate["guard_required_bytes"]),
        allocation_bytes=0,
    )
    preflight_path = output_root / "preflight.json"
    write_json_exclusive(
        preflight_path,
        {
            "schema": "token-reconstruction.trr-p01-method-preflight.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "source_truth_included": False,
            "selected_device": "cpu",
            "started_utc": started_utc,
            "implementation_commit": implementation_commit,
            "plan": file_record(plan_path),
            "prototype": file_record(prototype_path),
            "historical_lens": file_record(lens_path),
            "estimate": estimate,
            "guard": pre_guard,
            "numerics": {
                "deterministic_algorithms": True,
                "float32_distance": True,
                "tf32": False,
                "public_prefix_dtype": "bfloat16",
            },
        },
    )

    phases: list[dict[str, Any]] = []
    phase_started = time.perf_counter()
    table = PrototypeTable.load(
        prototype_path,
        expected_model_id=PUBLIC_MODEL_ID,
        expected_model_revision=PUBLIC_MODEL_REVISION,
        expected_cut_depth=CUT_DEPTH,
        expected_vocab_size=VOCAB_SIZE,
        expected_hidden_size=HIDDEN_SIZE,
    )
    if table.prototypes.dtype != torch.bfloat16:
        raise MethodQualificationError("public prototype table dtype changed")
    phases.append({"phase": "load_public_table", "elapsed_seconds": time.perf_counter() - phase_started})

    phase_started = time.perf_counter()
    model = load_public_model(device=device, model_path=args.model_path)
    prefix = ContiguousPublicPrefix(model, CUT_DEPTH).to(device).eval()
    if _module_device(prefix) != device:
        raise MethodQualificationError("public prefix device changed")
    if int(prefix.embed_tokens.num_embeddings) != VOCAB_SIZE:
        raise MethodQualificationError("public prefix vocabulary changed")
    phases.append({"phase": "load_public_model_and_prefix", "elapsed_seconds": time.perf_counter() - phase_started})
    post_model_guard = resource_guard(
        device=device,
        required_bytes=int(estimate["guard_required_bytes"]),
        allocation_bytes=0,
    )

    phase_started = time.perf_counter()
    with torch.inference_mode():
        observations = prefix.forward_full(input_ids.to(device=device))
    observations = _check_output(
        observations,
        (RECORDS, SEQUENCE_TOKENS, HIDDEN_SIZE),
        torch.bfloat16,
        "public probe observations",
    ).detach().cpu().contiguous()
    phases.append({"phase": "public_probe_observations", "elapsed_seconds": time.perf_counter() - phase_started})

    phase_started = time.perf_counter()
    with torch.inference_mode():
        base_cache = prefix.new_cache()
        cached_prefix = prefix.run_cached(
            input_ids[:, :PREFIX_LENGTH].to(device=device), base_cache, 0
        )
    cached_prefix = _check_output(
        cached_prefix,
        (RECORDS, PREFIX_LENGTH, HIDDEN_SIZE),
        torch.bfloat16,
        "public cached prefix",
    )
    cached_prefix_cpu = cached_prefix.detach().cpu().contiguous()
    full_prefix_cpu = observations[:, :PREFIX_LENGTH].detach().cpu().contiguous()
    prefix_difference = (cached_prefix_cpu.float() - full_prefix_cpu.float()).abs()
    _finite(prefix_difference, "full-versus-cached prefix difference")
    prefix_exact_equal = bool(torch.equal(cached_prefix_cpu, full_prefix_cpu))
    prefix_max_abs_difference = float(prefix_difference.max().item())
    if int(getattr(base_cache, "length", -1)) != PREFIX_LENGTH:
        raise MethodQualificationError("public prefix cache length changed")
    phases.append({"phase": "prepare_public_prefix_length_39", "elapsed_seconds": time.perf_counter() - phase_started})

    normalized = F.normalize(prefix.embed_tokens.weight.detach().float(), dim=-1)
    _finite(normalized, "normalized public input embedding table")
    lens = load_published_frozen_lens(lens_path, device=device)
    target = observations[:, QUALIFICATION_POSITION, :].to(device=device)

    phase_started = time.perf_counter()
    with torch.inference_mode():
        logits = lens(target, normalized).float()
    if tuple(logits.shape) != (RECORDS, VOCAB_SIZE):
        raise MethodQualificationError("historical lens vocabulary logits geometry changed")
    _finite(logits, "historical lens logits")
    top_scores, top_ids = torch.topk(
        logits, k=A1_TOP_K, dim=1, largest=True, sorted=True
    )
    top_scores = top_scores.detach().float()
    top_ids = top_ids.detach().to(torch.int32)
    candidate_ids = top_ids[:, :A2_BUDGET]
    candidate_scores = top_scores[:, :A2_BUDGET]
    _finite(top_scores, "historical A1 top-k scores")
    phases.append({"phase": "historical_a1_top512", "elapsed_seconds": time.perf_counter() - phase_started})

    phase_started = time.perf_counter()
    candidate_cache = _repeat_cache(base_cache, A2_BUDGET)
    with torch.inference_mode():
        candidate_output = prefix.run_cached(
            candidate_ids.to(device=device, dtype=torch.long).reshape(-1, 1),
            candidate_cache,
            QUALIFICATION_POSITION,
        )
    candidate_output = _check_output(
        candidate_output,
        (RECORDS * A2_BUDGET, 1, HIDDEN_SIZE),
        torch.bfloat16,
        "K256 candidate output",
    )[:, -1, :].reshape(RECORDS, A2_BUDGET, HIDDEN_SIZE).detach().cpu().contiguous()
    candidate_scores_direct = F.cosine_similarity(
        candidate_output.float(), observations[:, QUALIFICATION_POSITION, :].float()[:, None, :], dim=-1
    )
    _finite(candidate_scores_direct, "K256 direct-cosine scores")
    candidate_predictions = candidate_ids.gather(
        1, candidate_scores_direct.to(device="cpu").argmax(dim=1, keepdim=True)
    ).squeeze(1)
    if candidate_predictions.shape != (RECORDS,):
        raise MethodQualificationError("K256 winner geometry changed")
    phases.append({"phase": "k256_candidate_simulation_at_position_39", "elapsed_seconds": time.perf_counter() - phase_started})

    phase_started = time.perf_counter()
    reference_cache = _repeat_cache(base_cache, 1)
    with torch.inference_mode():
        reference_output = prefix.run_cached(
            torch.full((RECORDS, 1), REFERENCE_TOKEN, dtype=torch.long, device=device),
            reference_cache,
            QUALIFICATION_POSITION,
        )
    reference_hidden = _check_output(
        reference_output,
        (RECORDS, 1, HIDDEN_SIZE),
        torch.bfloat16,
        "reference-token output",
    )[:, -1, :].detach().cpu().contiguous()
    reference_prototype = table.prototypes[REFERENCE_TOKEN].float()
    reference_offset = reference_hidden.float() - reference_prototype
    _finite(reference_offset, "reference offset")
    if int(getattr(base_cache, "length", -1)) != PREFIX_LENGTH:
        raise MethodQualificationError("reference probe mutated persistent prefix cache")
    phases.append({"phase": "reference_token_220_probe_at_position_39", "elapsed_seconds": time.perf_counter() - phase_started})

    phase_started = time.perf_counter()
    nearest: dict[str, Any] = {}
    for metric in ("cosine", "l2"):
        result = table.nearest(
            observations[:, QUALIFICATION_POSITION, :],
            metric=metric,
            query_chunk_size=256,
            prototype_chunk_size=8192,
        )
        _finite(result.scores, f"full-vocabulary {metric} scores")
        _finite(result.margins, f"full-vocabulary {metric} margins")
        nearest[metric] = result
    phases.append({"phase": "full_vocabulary_nearest_lookup", "elapsed_seconds": time.perf_counter() - phase_started})

    output_tensors: dict[str, torch.Tensor] = {
        "public_probe_input_ids": input_ids.to(torch.int32),
        "public_probe_observations": observations,
        "cached_prefix_observations": cached_prefix_cpu,
        "a1_top512_ids": top_ids.cpu().contiguous(),
        "a1_top512_scores": top_scores.cpu().contiguous(),
        "candidate_ids_k256": candidate_ids.cpu().contiguous(),
        "candidate_scores_a1_k256": candidate_scores.cpu().contiguous(),
        "candidate_hidden_k256": candidate_output,
        "candidate_scores_direct_cosine": candidate_scores_direct.cpu().contiguous(),
        "candidate_predictions_k256": candidate_predictions.to(torch.int32),
        "reference_hidden_token220": reference_hidden,
        "reference_offset_token220": reference_offset,
    }
    for metric, result in nearest.items():
        output_tensors[f"nearest_{metric}_predictions"] = result.predictions.to(torch.int32)
        output_tensors[f"nearest_{metric}_scores"] = result.scores.float()
        output_tensors[f"nearest_{metric}_margins"] = result.margins.float()
    for name, value in output_tensors.items():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise MethodQualificationError(f"qualification output is not CPU resident: {name}")
        _finite(value.float(), name)
    post_cell_guard = resource_guard(
        device=device,
        required_bytes=int(estimate["guard_required_bytes"]),
        allocation_bytes=0,
    )

    output_path = output_root / "method_qualification.safetensors"
    require_create_only_file(output_path)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in output_tensors.items()},
        output_path,
        metadata={
            "schema": OUTPUT_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": "false",
            "source_truth_included": "false",
            "selected_device": "cpu",
            "policy_id": HISTORICAL_POLICY_ID,
        },
    )
    output_record = file_record(output_path)
    phase_timing = phases
    evidence_path = output_root / "qualification_evidence.json"
    require_create_only_file(evidence_path)
    evidence = {
        "schema": QUALIFICATION_SCHEMA,
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": False,
        "status": "CPU_METHOD_CELL_QUALIFIED",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "implementation_commit": implementation_commit,
        "selected_device": "cpu",
        "model": {
            "id": PUBLIC_MODEL_ID,
            "revision": PUBLIC_MODEL_REVISION,
            "cut_depth": CUT_DEPTH,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
        },
        "policy": {
            "id": HISTORICAL_POLICY_ID,
            "lens_sha256": lens_hash,
            "a1_top_k": A1_TOP_K,
            "a1_chunk": 256,
            "a2_budget": A2_BUDGET,
            "score": "direct_cosine",
            "terminal_action": "always_commit_winner",
            "tie_behavior": "first candidate in published torch.topk order via argmax",
        },
        "public_probe": {
            "probe_token_source": "qualification.probe_token_ids from frozen public plan",
            "probe_count": len(probe_ids),
            "probe_id_digest": digest_tensor(torch.tensor(probe_ids, dtype=torch.int64)),
            "input_ids": _tensor_record(input_ids.to(torch.int32)),
            "observations": _tensor_record(observations),
            "cached_prefix_diagnostic": {
                "full_prefix_shape": [RECORDS, PREFIX_LENGTH, HIDDEN_SIZE],
                "cached_prefix_shape": [RECORDS, PREFIX_LENGTH, HIDDEN_SIZE],
                "full_vs_cached_exact_equal": prefix_exact_equal,
                "full_vs_cached_max_abs_difference": prefix_max_abs_difference,
                "comparison": "diagnostic_only; cached length-39 execution remains the native qualification path",
            },
            "record_count": RECORDS,
            "sequence_tokens": SEQUENCE_TOKENS,
            "prefix_length": PREFIX_LENGTH,
            "qualification_position": QUALIFICATION_POSITION,
        },
        "candidate_cell": {
            "record_batch_size": RECORD_BATCH_SIZE,
            "candidate_budget": A2_BUDGET,
            "candidate_batch_rows": RECORDS * A2_BUDGET,
            "candidate_simulations": RECORDS * A2_BUDGET,
            "candidate_cache_commits": RECORDS * A2_BUDGET,
            "persistent_prefix_cache_commits": RECORDS * PREFIX_LENGTH,
            "persistent_cache_length_after_probes": int(getattr(base_cache, "length", -1)),
            "candidate_ids": _tensor_record(candidate_ids),
            "candidate_hidden": _tensor_record(candidate_output),
            "direct_scores": _tensor_record(candidate_scores_direct),
            "winner_ids": _tensor_record(candidate_predictions.to(torch.int32)),
        },
        "reference_probe": {
            "token_id": REFERENCE_TOKEN,
            "evaluations": RECORDS,
            "cache_commits": RECORDS,
            "prefix_length_before_probe": PREFIX_LENGTH,
            "persistent_cache_unchanged": int(getattr(base_cache, "length", -1)) == PREFIX_LENGTH,
            "hidden": _tensor_record(reference_hidden),
            "offset": _tensor_record(reference_offset),
        },
        "full_vocabulary_lookup": {
            "vocab_size": VOCAB_SIZE,
            "query_rows": RECORDS,
            "prototype_chunk_size": 8192,
            "query_chunk_size": 256,
            "metrics": {
                metric: {
                    "predictions": _tensor_record(result.predictions.to(torch.int32)),
                    "scores": _tensor_record(result.scores),
                    "margins": _tensor_record(result.margins),
                }
                for metric, result in nearest.items()
            },
        },
        "counts": {
            "a1_forward_calls": 1,
            "public_prefix_calls": 4,
            "public_prefix_token_evaluations": (
                RECORDS * SEQUENCE_TOKENS
                + RECORDS * PREFIX_LENGTH
                + RECORDS * A2_BUDGET
                + RECORDS
            ),
            "candidate_simulations": RECORDS * A2_BUDGET,
            "reference_evaluations": RECORDS,
            "persistent_cache_commits": RECORDS * PREFIX_LENGTH,
            "candidate_cache_commits": RECORDS * A2_BUDGET,
            "reference_cache_commits": RECORDS,
        },
        "timing": {
            "phases": phase_timing,
            "total_seconds": time.perf_counter() - started,
        },
        "resource": {
            "preflight_estimate": estimate,
            "pre_model_guard": pre_guard,
            "post_model_guard": post_model_guard,
            "post_cell_guard": post_cell_guard,
            "peak_memory": peak_memory(device),
        },
        "artifacts": {
            "plan": file_record(plan_path),
            "prototype": file_record(prototype_path),
            "historical_lens": file_record(lens_path),
            "preflight": file_record(preflight_path),
            "output": output_record,
        },
        "source_files": _source_files(lens_path),
        "runtime": environment_record(device),
    }
    write_json_exclusive(evidence_path, evidence)
    print(
        {
            "status": evidence["status"],
            "output": str(output_path),
            "evidence": str(evidence_path),
            "candidate_simulations": RECORDS * A2_BUDGET,
            "truth_opened": False,
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except MethodQualificationError as exc:
        print(f"qualify_methods: {exc}", file=sys.stderr)
        raise SystemExit(2)
