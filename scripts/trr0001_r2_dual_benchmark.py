#!/usr/bin/env python3
"""Retrospective six-cell dual-benchmark backfill for TRR-0001-R2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import transformers

from token_reconstruction.dual_benchmark import (
    BOS_TOKEN_ID,
    METHOD_IDS,
    SETUP_IDS,
    causal_k16,
    paired_record_differences,
    propose_k16,
    score_predictions,
    validate_observations,
)
from token_reconstruction.experiment_runtime import seed_everything
from token_reconstruction.inverse import load_inverse
from token_reconstruction.metrics import bootstrap_mean


MODEL_SPEC = {
    "id": "meta-llama/Llama-3.2-1B-Instruct",
    "revision": "9213176726f574b556790deb65791e0c5aa438b6",
    "prefix_layers": [0, 1, 2, 3],
    "dtype": "bfloat16",
    "attention_implementation": "sdpa",
    "local_files_only": True,
}
NEW_SETUP = SETUP_IDS[0]
OLD_SETUP = SETUP_IDS[1]
DIRECT = METHOD_IDS[0]
CAUSAL = METHOD_IDS[1]
STRICT = METHOD_IDS[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--new-input-root", type=Path, required=True)
    parser.add_argument("--new-truth-jsonl", type=Path, required=True)
    parser.add_argument("--r1-direct-jsonl", type=Path, required=True)
    parser.add_argument("--r1-causal-jsonl", type=Path, required=True)
    parser.add_argument("--old-native-json", type=Path, required=True)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(f"JSONL row {line_number} is not an object")
                rows.append(row)
    return rows


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def new_inputs(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = root / "observations" / "unavailable_target_lora_cut4.safetensors"
    state = load_file(path, device="cpu")
    if set(state) != {"activations"}:
        raise RuntimeError("clean cut-4 observation fields changed")
    observations = state["activations"].contiguous()
    if observations.shape != (64, 40, 2048):
        raise RuntimeError("clean cut-4 observation geometry changed")
    attention_mask = torch.ones((64, 40), dtype=torch.long)
    position_ids = torch.arange(40, dtype=torch.long).view(1, -1).expand(64, -1)
    validate_observations(observations, attention_mask, position_ids)
    return observations, attention_mask, position_ids


def historical_inputs(
    historical_root: Path,
    source300: Any,
) -> tuple[dict[str, Any], list[Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    config_path = (
        historical_root / "config" / "a1_a2_source300_static_20260809.json"
    )
    config = source300.load_config(config_path)
    source_path = source300.resolve_inside_ersoy(config["source"]["path"])
    captures = source300.load_source_payload(source_path, config)
    observations = torch.cat(
        [capture.activation.detach().cpu() for capture in captures],
        dim=0,
    ).contiguous()
    attention_mask = torch.cat(
        [capture.attention_mask.detach().cpu().to(torch.long) for capture in captures],
        dim=0,
    ).contiguous()
    position_ids = torch.cat(
        [capture.position_ids.detach().cpu().to(torch.long) for capture in captures],
        dim=0,
    ).contiguous()
    if observations.shape != (128, 128, 2048):
        raise RuntimeError("historical observation geometry changed")
    validate_observations(observations, attention_mask, position_ids)
    return config, captures, observations, attention_mask, position_ids


@torch.inference_mode()
def strict_decode(
    *,
    reference: Any,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    lens: torch.nn.Module,
    embeddings: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    validate_observations(observations, attention_mask, position_ids)
    records, positions, hidden = observations.shape
    if hidden != 2048:
        raise RuntimeError("strict-BOS hidden size changed")
    torch.cuda.synchronize(device)
    proposal_started = time.perf_counter()
    candidates, confidence = reference.rank_topk(
        observations.reshape(-1, hidden),
        lens=lens,
        normalized_embeddings=embeddings,
    )
    torch.cuda.synchronize(device)
    proposal_seconds = time.perf_counter() - proposal_started
    candidates = candidates.reshape(records, positions, 512)
    confidence = confidence.reshape(records, positions)

    results = []
    torch.cuda.synchronize(device)
    selection_started = time.perf_counter()
    for row_index in range(records):
        row = reference.PassiveRow(
            row_index=row_index,
            activation=observations[row_index],
            attention_mask=attention_mask[row_index],
            position_ids=position_ids[row_index],
        )
        results.append(
            reference.decode_teacher_row(
                row,
                candidates=candidates[row_index],
                a1_confidence=confidence[row_index],
                precut=precut,
                device=device,
            )
        )
    torch.cuda.synchronize(device)
    selection_seconds = time.perf_counter() - selection_started
    predictions = torch.stack([result.token_ids for result in results])
    routes = torch.stack([result.route_codes for result in results])
    return (
        predictions,
        candidates.to(torch.long),
        routes,
        {
            "proposal_seconds": proposal_seconds,
            "selection_seconds": selection_seconds,
            "compute_seconds": proposal_seconds + selection_seconds,
            "candidate_simulations": sum(
                result.candidate_simulations for result in results
            ),
        },
    )


def load_new_truth(
    path: Path,
    expected_records: int,
) -> tuple[torch.Tensor, list[str]]:
    rows = load_jsonl(path)
    if len(rows) != expected_records:
        raise RuntimeError("clean truth record count changed")
    record_ids = [str(row["record_id"]) for row in rows]
    truth = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long)
    if truth.shape != (64, 40) or not truth[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise RuntimeError("clean truth geometry or BOS changed")
    return truth, record_ids


def load_old_truth(
    source300: Any,
    captures: list[Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, list[str]]:
    tokenizer = source300.load_tokenizer(config)
    rows = source300.reconstruct_rows(captures, tokenizer, config)
    truth = torch.stack([row.truth_ids for row in rows]).to(torch.long)
    record_ids = [f"source300-row-{row.row_index:03d}" for row in rows]
    if truth.shape != (128, 128) or not truth[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise RuntimeError("historical truth geometry or BOS changed")
    return truth, record_ids


def frozen_r1_predictions(
    path: Path,
    record_ids: list[str],
) -> torch.Tensor:
    selected = [
        row
        for row in load_jsonl(path)
        if row["condition"] == "unavailable_target_lora"
        and int(row["cut_depth"]) == 4
    ]
    if len(selected) != 64:
        raise RuntimeError("R1 frozen cut-4 row count changed")
    by_id = {str(row["record_id"]): row for row in selected}
    if set(by_id) != set(record_ids):
        raise RuntimeError("R1 frozen record IDs changed")
    predictions = torch.full((64, 40), -1, dtype=torch.long)
    predictions[:, 0] = BOS_TOKEN_ID
    for index, record_id in enumerate(record_ids):
        values = by_id[record_id]["prediction_tokens"]
        if len(values) != 39:
            raise RuntimeError("R1 frozen prediction length changed")
        predictions[index, 1:] = torch.tensor(values, dtype=torch.long)
    return predictions


def matrix_cell(
    *,
    status: str,
    metrics: dict[str, Any],
    per_record: list[dict[str, Any]],
    timing: dict[str, Any],
    port_differences: list[str],
    candidates: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "metrics": metrics,
        "per_record": per_record,
        "timing": timing,
        "port_differences": port_differences,
        "candidate_budget": candidates,
    }


def main() -> int:
    args = parse_args()
    started_utc = utc_now()
    seed_everything(1729)
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    prediction_path = args.prediction_artifact
    output_path = args.output
    if prediction_path.exists() or output_path.exists():
        raise RuntimeError("R2 output paths are create-only")

    reference_path = repository_root / "reference" / "strict_bos" / "round001_teacher.py"
    reference = import_path("trr_r2_strict_reference", reference_path)
    source_path = (
        historical_root / "scripts" / "score_a1_a2_source300_20260809.py"
    )
    source300 = import_path("trr_r2_source300", source_path)

    new_observations, new_mask, new_positions = new_inputs(
        args.new_input_root.resolve(strict=True)
    )
    (
        old_config,
        old_captures,
        old_observations,
        old_mask,
        old_positions,
    ) = historical_inputs(historical_root, source300)

    identity_path = (
        historical_root
        / "research"
        / "adaptive_a1_a2_strict_bos_20260817_goal_01a00b08"
        / "audit"
        / "AUDIT-0004-public-teacher-identity-v2.json"
    )
    lens_path = historical_root / "inversion_20260730" / "out" / "lens_alpaca.pt"
    identity = load_json(identity_path)
    precut, strict_lens, embeddings, device, observed_identity = (
        reference.load_public_teacher(
            MODEL_SPEC,
            identity,
            lens_path=lens_path,
        )
    )
    inverse_path = args.new_input_root / "inverses" / "cut4.safetensors"
    inverse = load_inverse(inverse_path, hidden_size=2048, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    new_direct, new_k16, new_proposal_scores, new_proposal_seconds = propose_k16(
        observations=new_observations,
        attention_mask=new_mask,
        inverse=inverse,
        embedding_table=embeddings,
    )
    new_causal, new_selection_scores, new_selection_seconds, new_simulations = (
        causal_k16(
            observations=new_observations,
            attention_mask=new_mask,
            position_ids=new_positions,
            candidates=new_k16,
            precut=precut,
            device=device,
        )
    )
    new_strict, new_k512, new_routes, new_strict_timing = strict_decode(
        reference=reference,
        observations=new_observations,
        attention_mask=new_mask,
        position_ids=new_positions,
        lens=strict_lens,
        embeddings=embeddings,
        precut=precut,
        device=device,
    )

    old_direct, old_k16, old_proposal_scores, old_proposal_seconds = propose_k16(
        observations=old_observations,
        attention_mask=old_mask,
        inverse=inverse,
        embedding_table=embeddings,
    )
    old_causal, old_selection_scores, old_selection_seconds, old_simulations = (
        causal_k16(
            observations=old_observations,
            attention_mask=old_mask,
            position_ids=old_positions,
            candidates=old_k16,
            precut=precut,
            device=device,
        )
    )
    old_strict, old_k512, old_routes, old_strict_timing = strict_decode(
        reference=reference,
        observations=old_observations,
        attention_mask=old_mask,
        position_ids=old_positions,
        lens=strict_lens,
        embeddings=embeddings,
        precut=precut,
        device=device,
    )
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "new.direct.predictions": new_direct.to(torch.int32).contiguous(),
            "new.causal.predictions": new_causal.to(torch.int32).contiguous(),
            "new.strict.predictions": new_strict.to(torch.int32).contiguous(),
            "new.k16.candidates": new_k16.to(torch.int32).contiguous(),
            "new.k512.candidates": new_k512.to(torch.int32).contiguous(),
            "new.strict.routes": new_routes.to(torch.int8).contiguous(),
            "old.direct.predictions": old_direct.to(torch.int32).contiguous(),
            "old.causal.predictions": old_causal.to(torch.int32).contiguous(),
            "old.strict.predictions": old_strict.to(torch.int32).contiguous(),
            "old.k16.candidates": old_k16.to(torch.int32).contiguous(),
            "old.k512.candidates": old_k512.to(torch.int32).contiguous(),
            "old.strict.routes": old_routes.to(torch.int8).contiguous(),
        },
        prediction_path,
        metadata={
            "schema": "token-reconstruction.trr0001-r2-dual-predictions.v1",
            "truth_status": "already-open-retrospective-backfill",
            "methods": ",".join(METHOD_IDS),
            "setups": ",".join(SETUP_IDS),
        },
    )
    prediction_record_before_truth = file_record(prediction_path)

    new_truth, new_ids = load_new_truth(args.new_truth_jsonl, 64)
    old_truth, old_ids = load_old_truth(source300, old_captures, old_config)

    new_direct_metrics, new_direct_rows = score_predictions(
        predictions=new_direct,
        truth=new_truth,
        attention_mask=new_mask,
        candidates=new_k16,
        record_ids=new_ids,
    )
    new_causal_metrics, new_causal_rows = score_predictions(
        predictions=new_causal,
        truth=new_truth,
        attention_mask=new_mask,
        candidates=new_k16,
        record_ids=new_ids,
    )
    new_strict_metrics, new_strict_rows = score_predictions(
        predictions=new_strict,
        truth=new_truth,
        attention_mask=new_mask,
        candidates=new_k512,
        record_ids=new_ids,
    )
    old_direct_metrics, old_direct_rows = score_predictions(
        predictions=old_direct,
        truth=old_truth,
        attention_mask=old_mask,
        candidates=old_k16,
        record_ids=old_ids,
    )
    old_causal_metrics, old_causal_rows = score_predictions(
        predictions=old_causal,
        truth=old_truth,
        attention_mask=old_mask,
        candidates=old_k16,
        record_ids=old_ids,
    )
    old_strict_metrics, old_strict_rows = score_predictions(
        predictions=old_strict,
        truth=old_truth,
        attention_mask=old_mask,
        candidates=old_k512,
        record_ids=old_ids,
    )

    frozen_new_direct = frozen_r1_predictions(args.r1_direct_jsonl, new_ids)
    frozen_new_causal = frozen_r1_predictions(args.r1_causal_jsonl, new_ids)
    new_reproduction = {
        "direct_prediction_mismatches": int(
            new_direct.ne(frozen_new_direct).sum().item()
        ),
        "causal_prediction_mismatches": int(
            new_causal.ne(frozen_new_causal).sum().item()
        ),
    }
    if any(new_reproduction.values()):
        raise RuntimeError(f"clean native reproduction failed: {new_reproduction}")

    old_native = load_json(args.old_native_json)
    old_expected = old_native["row_serial_adaptive_a1_a2"]["metrics_full"]
    old_native_checks = {
        "scored_tokens": old_strict_metrics["scored_tokens"]
        == old_expected["unknown_tokens"],
        "covered_tokens": old_strict_metrics["covered_tokens"]
        == old_expected["unknown_covered_tokens"],
        "correct_tokens": old_strict_metrics["correct_tokens"]
        == old_expected["unknown_correct_covered_tokens"],
        "token_accuracy": old_strict_metrics["token_accuracy"]
        == old_expected["unknown_end_to_end_accuracy"],
        "selective_accuracy": old_strict_metrics["selective_accuracy"]
        == old_expected["unknown_selective_accuracy"],
        "exact_records": old_strict_metrics["exact_records"]
        == old_expected["whole_row_exact_count"],
        "candidate_simulations": old_strict_timing["candidate_simulations"]
        == old_expected["candidate_simulations"],
    }
    if not all(old_native_checks.values()):
        raise RuntimeError(f"historical native reproduction failed: {old_native_checks}")

    new_direct_timing = {
        "proposal_seconds": new_proposal_seconds,
        "selection_seconds": 0.0,
        "compute_seconds": new_proposal_seconds,
        "candidate_simulations": 0,
    }
    new_causal_timing = {
        "proposal_seconds": new_proposal_seconds,
        "selection_seconds": new_selection_seconds,
        "compute_seconds": new_proposal_seconds + new_selection_seconds,
        "candidate_simulations": new_simulations,
    }
    old_direct_timing = {
        "proposal_seconds": old_proposal_seconds,
        "selection_seconds": 0.0,
        "compute_seconds": old_proposal_seconds,
        "candidate_simulations": 0,
    }
    old_causal_timing = {
        "proposal_seconds": old_proposal_seconds,
        "selection_seconds": old_selection_seconds,
        "compute_seconds": old_proposal_seconds + old_selection_seconds,
        "candidate_simulations": old_simulations,
    }

    matrix = {
        NEW_SETUP: {
            DIRECT: matrix_cell(
                status="native_blind_result_reproduced_retrospectively",
                metrics=new_direct_metrics,
                per_record=new_direct_rows,
                timing=new_direct_timing,
                port_differences=[],
                candidates=16,
            ),
            CAUSAL: matrix_cell(
                status="native_blind_result_reproduced_retrospectively",
                metrics=new_causal_metrics,
                per_record=new_causal_rows,
                timing=new_causal_timing,
                port_differences=[],
                candidates=16,
            ),
            STRICT: matrix_cell(
                status="benchmark_compatible_port_retrospective",
                metrics=new_strict_metrics,
                per_record=new_strict_rows,
                timing=new_strict_timing,
                port_differences=[
                    "record loop bound changed from 128 to 64",
                    "position loop bound changed from right-padded 128 to dense 40",
                    "tensor packing changed; decode_teacher_row and all method constants are unchanged",
                ],
                candidates=512,
            ),
        },
        OLD_SETUP: {
            DIRECT: matrix_cell(
                status="benchmark_compatible_port_retrospective",
                metrics=old_direct_metrics,
                per_record=old_direct_rows,
                timing=old_direct_timing,
                port_differences=[
                    "right-padding mask selects variable valid positions",
                    "record-serial causal-compatible tensor packing replaces dense 64x40 packing",
                    "inverse state, K16 proposals, candidate ordering, and direct rank-1 rule are unchanged",
                ],
                candidates=16,
            ),
            CAUSAL: matrix_cell(
                status="benchmark_compatible_port_retrospective",
                metrics=old_causal_metrics,
                per_record=old_causal_rows,
                timing=old_causal_timing,
                port_differences=[
                    "right-padding mask selects variable valid positions",
                    "record-serial execution replaces 16-record batches",
                    "inverse state, K16 candidates, cosine score, cache semantics, and greedy reconstructed-prefix rule are unchanged",
                ],
                candidates=16,
            ),
            STRICT: matrix_cell(
                status="exact_native_result_independently_reproduced",
                metrics=old_strict_metrics,
                per_record=old_strict_rows,
                timing=old_strict_timing,
                port_differences=[],
                candidates=512,
            ),
        },
    }

    comparisons = {}
    for setup, rows_by_method in (
        (
            NEW_SETUP,
            {
                DIRECT: new_direct_rows,
                CAUSAL: new_causal_rows,
                STRICT: new_strict_rows,
            },
        ),
        (
            OLD_SETUP,
            {
                DIRECT: old_direct_rows,
                CAUSAL: old_causal_rows,
                STRICT: old_strict_rows,
            },
        ),
    ):
        comparisons[setup] = {
            "causal_minus_direct": bootstrap_mean(
                paired_record_differences(
                    rows_by_method[CAUSAL],
                    rows_by_method[DIRECT],
                ),
                draws=10000,
                seed=2401,
            ),
            "strict_minus_direct": bootstrap_mean(
                paired_record_differences(
                    rows_by_method[STRICT],
                    rows_by_method[DIRECT],
                ),
                draws=10000,
                seed=2402,
            ),
            "strict_minus_causal": bootstrap_mean(
                paired_record_differences(
                    rows_by_method[STRICT],
                    rows_by_method[CAUSAL],
                ),
                draws=10000,
                seed=2403,
            ),
        }

    payload = {
        "schema": "token-reconstruction.trr0001-r2-dual-benchmark-matrix.v1",
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R2",
        "status": "RETROSPECTIVE_COMPLETE_MATRIX",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "truth_status": {
            NEW_SETUP: "already opened by completed TRR-0001-R1 scoring",
            OLD_SETUP: "already opened historical source-300 benchmark",
            "claim_limit": "comparability backfill, not fresh confirmation",
            "prediction_artifact_written_before_this_runner_loaded_truth": True,
        },
        "method_order": list(METHOD_IDS),
        "setup_order": list(SETUP_IDS),
        "matrix_complete": all(
            set(matrix[setup]) == set(METHOD_IDS) for setup in SETUP_IDS
        ),
        "matrix": matrix,
        "paired_record_bootstrap": comparisons,
        "semantic_checks": {
            "clean_native_prediction_reproduction": new_reproduction,
            "historical_strict_native_aggregate_reproduction": old_native_checks,
        },
        "cost_scope": {
            "timing_session": "single R2 backfill process",
            "direct_and_causal_share_identical_measured_proposal_time_per_setup": True,
            "model_load_excluded": True,
            "file_io_excluded": True,
            "cuda_synchronized_at_boundaries": True,
            "hardware": torch.cuda.get_device_name(device),
            "peak_cuda_memory_allocated_bytes": peak_memory_bytes,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "pid": os.getpid(),
        },
        "public_teacher_identity": observed_identity,
        "artifacts": {
            "prediction_freeze": prediction_record_before_truth,
            "new_observation": file_record(
                args.new_input_root
                / "observations"
                / "unavailable_target_lora_cut4.safetensors"
            ),
            "new_inverse": file_record(inverse_path),
            "new_truth": file_record(args.new_truth_jsonl),
            "old_source": file_record(
                source300.resolve_inside_ersoy(old_config["source"]["path"])
            ),
            "old_native_rerun": file_record(args.old_native_json),
            "strict_lens": file_record(lens_path),
            "strict_reference_source": file_record(reference_path),
            "dual_primitive_source": file_record(
                repository_root
                / "src"
                / "token_reconstruction"
                / "dual_benchmark.py"
            ),
            "runner_source": file_record(Path(__file__)),
        },
    }
    if not payload["matrix_complete"]:
        raise RuntimeError("dual-benchmark matrix is incomplete")
    write_json_exclusive(output_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "matrix_complete": payload["matrix_complete"],
                "output": str(output_path),
                "prediction_artifact": str(prediction_path),
                "new_accuracy": {
                    method: matrix[NEW_SETUP][method]["metrics"]["token_accuracy"]
                    for method in METHOD_IDS
                },
                "old_accuracy": {
                    method: matrix[OLD_SETUP][method]["metrics"]["token_accuracy"]
                    for method in METHOD_IDS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
