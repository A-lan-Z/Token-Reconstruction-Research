#!/usr/bin/env python3
"""Evaluator-only construction of matched and shifted public interfaces.

The command reads the evaluator-private panel truth and writes two separately
named, condition-free public arm directories.  The private condition mapping
and truth never enter either arm directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from token_reconstruction.public_prefix import ContiguousPublicPrefix  # noqa: E402
from token_reconstruction.trr_p01 import PrototypeTable  # noqa: E402
from common import (  # noqa: E402
    BOS_TOKEN_ID,
    CONFIG_SCHEMA,
    CONDITIONS,
    CUT_DEPTH,
    HIDDEN_SIZE,
    METRICS,
    MODEL_ID,
    MODEL_REVISION,
    OBSERVATION_INDEX_SCHEMA,
    OBSERVATION_SCHEMA,
    PREDICTION_SCHEMA,
    SCORED_TOKENS,
    SEQUENCE_TOKENS,
    TASK_ID,
    artifact_entry,
    command_record,
    digest_tensor,
    environment_record,
    file_record,
    load_json,
    load_public_model,
    load_target_model,
    peak_memory,
    position_digest,
    require_create_only_directory,
    require_create_only_file,
    seed_everything,
    sha256_file,
    mask_digest,
    utc_now,
    validate_public_plan,
    write_json_exclusive,
)


PRIVATE_TRUTH_SCHEMA = "token-reconstruction.trr-p01-private-truth.v1"
EVALUATOR_SCHEMA = "token-reconstruction.trr-p01-evaluator-evidence.v1"
TARGET_CONFIG_SHA256 = "7510055506497971937a3b247c853e664fdc1b1bbeece4cafc03107fa5e6fae7"
TARGET_MODEL_SHA256 = "389c73748a00a8a006a4a4a26fa473319676c25672aa188f8337981cd0cc8850"

# The blind prediction matrix is fixed before either arm is reconstructed or
# any private truth is opened.  ``prediction_arms`` is explicit because the
# historical controls intentionally expose cosine only; declaring the
# Cartesian product would incorrectly require historical L2 predictions.
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
PREDICTION_METHODS = (
    "boundary",
    "raw_embedding",
    "reference_corrected",
    "historical_a1",
    "historical_a1_a2_port",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--public-model-path", type=Path, default=None)
    parser.add_argument("--target-model-path", type=Path, default=None)
    parser.add_argument("--record-batch-size", type=int, default=8)
    return parser.parse_args()


def _load_private_truth(panel_root: Path) -> tuple[dict[str, Any], torch.Tensor, Path]:
    manifest_path = panel_root / "panel_manifest.json"
    truth_path = panel_root / "private_truth.safetensors"
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema") != "token-reconstruction.trr-p01-panel.v1"
        or manifest.get("task_id") != TASK_ID
        or manifest.get("truth_opened") is not False
        or manifest.get("source_truth_included") is not True
        or manifest.get("record_order") != [f"p01-r{i:04d}" for i in range(1, 17)]
    ):
        raise RuntimeError("panel manifest identity changed")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 16:
        raise RuntimeError("private panel record geometry changed")
    style_counts: dict[str, int] = {}
    for row in records:
        if set(row) != {"record_id", "style", "dataset_index", "text_sha256", "source_token_count"}:
            raise RuntimeError("private panel source fields changed")
        style_counts[str(row["style"])] = style_counts.get(str(row["style"]), 0) + 1
    if style_counts != {
        "prose": 4,
        "code": 4,
        "numeric_plus_punctuation": 4,
        "unicode_plus_instruction": 4,
    }:
        raise RuntimeError("private panel style strata changed")
    expected_truth = manifest.get("private_truth")
    if not isinstance(expected_truth, dict) or expected_truth.get("path") != "private_truth.safetensors":
        raise RuntimeError("private truth manifest entry changed")
    if truth_path.is_symlink() or not truth_path.is_file():
        raise RuntimeError("private truth file unavailable")
    if truth_path.stat().st_size != int(expected_truth["bytes"]) or sha256_file(truth_path) != expected_truth["sha256"]:
        raise RuntimeError("private truth file hash changed")
    with safe_open(truth_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"input_ids"}:
            raise RuntimeError("private truth tensor fields changed")
        if handle.metadata() != {
            "schema": PRIVATE_TRUTH_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": "false",
            "source_truth_included": "true",
        }:
            raise RuntimeError("private truth metadata changed")
        truth = handle.get_tensor("input_ids")
    if truth.dtype != torch.int64 or tuple(truth.shape) != (16, SEQUENCE_TOKENS):
        raise RuntimeError("private truth geometry changed")
    if not torch.equal(truth[:, 0], torch.full((16,), BOS_TOKEN_ID, dtype=torch.int64)):
        raise RuntimeError("private truth BOS changed")
    if truth[:, 1:].min().item() < 0 or truth[:, 1:].max().item() >= 128256:
        raise RuntimeError("private truth vocabulary changed")
    return manifest, truth, truth_path


def _collect_observations(
    model: Any,
    truth: torch.Tensor,
    *,
    device: torch.device,
    record_batch_size: int,
) -> torch.Tensor:
    if record_batch_size <= 0 or record_batch_size > 16:
        raise RuntimeError("record batch size is outside the frozen bound")
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, truth.shape[0], record_batch_size):
            tokens = truth[start : start + record_batch_size].to(device)
            output = model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                output_hidden_states=True,
                use_cache=False,
            )
            if output.hidden_states is None or len(output.hidden_states) <= CUT_DEPTH:
                raise RuntimeError("model did not return the cut activation")
            activation = output.hidden_states[CUT_DEPTH]
            if tuple(activation.shape) != (tokens.shape[0], SEQUENCE_TOKENS, HIDDEN_SIZE):
                raise RuntimeError("observation geometry changed")
            values.append(activation.detach().to(device="cpu", dtype=torch.bfloat16))
            del output
    result = torch.cat(values, dim=0).contiguous()
    if tuple(result.shape) != (16, SEQUENCE_TOKENS, HIDDEN_SIZE) or not torch.isfinite(result).all().item():
        raise RuntimeError("observation values are invalid")
    return result


def _write_arm(
    arm_root: Path,
    observations: torch.Tensor,
    *,
    plan: dict[str, Any],
    prototype_path: Path,
    prototype_digest: str,
) -> dict[str, Any]:
    arm_root.mkdir()
    observation_path = arm_root / "observations.safetensors"
    require_create_only_file(observation_path)
    save_file(
        {"activations": observations},
        observation_path,
        metadata={
            "schema": OBSERVATION_SCHEMA,
            "opaque_records": "true",
            "source_truth_included": "false",
        },
    )
    rows = []
    for record_id, row in zip(
        [f"p01-r{i:04d}" for i in range(1, 17)], observations
    ):
        rows.append(
            {
                "record_id": record_id,
                "sequence_length": SEQUENCE_TOKENS,
                "mask_digest": mask_digest(),
                "position_digest": position_digest(),
                "observation_digest": digest_tensor(row),
            }
        )
    index_path = arm_root / "observation_index.json"
    observation_entry = artifact_entry(observation_path, relative_to=arm_root)
    write_json_exclusive(
        index_path,
        {
            "schema": OBSERVATION_INDEX_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": False,
            "source_material_included": False,
            "geometry": {
                "records": 16,
                "sequence_tokens": SEQUENCE_TOKENS,
                "scored_tokens": SCORED_TOKENS,
                "hidden_size": HIDDEN_SIZE,
            },
            "records": rows,
            "observation": observation_entry,
        },
    )
    config_path = arm_root / "sanitized_config.json"
    write_json_exclusive(
        config_path,
        {
            "schema": CONFIG_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": False,
            "source_truth_included": False,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "hidden_size": HIDDEN_SIZE,
                "vocab_size": 128256,
                "cut_depth": CUT_DEPTH,
                "bos_token_id": BOS_TOKEN_ID,
                "dtype": "bfloat16",
                "attention_implementation": "sdpa",
            },
            "geometry": {
                "records": 16,
                "sequence_tokens": SEQUENCE_TOKENS,
                "scored_tokens": SCORED_TOKENS,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
            },
            "record_order": [f"p01-r{i:04d}" for i in range(1, 17)],
            "metric_order": list(METRICS),
            # Keep the general method list for human-readable configuration,
            # and bind the actual non-Cartesian arm matrix separately.
            "methods": list(PREDICTION_METHODS),
            "prototype": {
                "path": str(prototype_path),
                "bytes": prototype_path.stat().st_size,
                "sha256": prototype_digest,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "cut_depth": CUT_DEPTH,
                "vocab_size": 128256,
                "hidden_size": HIDDEN_SIZE,
            },
            "reference_correction": {
                "status": "predeclared_after_static_evidence",
                "reference_token_id": 220,
                "candidate_simulations": 0,
                "prefix_rule": "already committed reconstructed prefix only",
            },
            "prediction_arms": list(PREDICTION_ARMS),
            "prediction_policy": {
                "status": "all_declared_arms_must_be_serialized_before_truth",
                "historical_metrics": ["cosine"],
            },
            "execution": {
                "seed": 1701,
                "record_batch_size": 8,
                "query_chunk_size": 256,
                "prototype_chunk_size": 8192,
                "stopping": "all 39 post-BOS positions",
                "attention_mask": "all ones",
                "position_ids": "0..39",
                "condition_label_visible_to_attack": False,
                "target_callable_visible_to_attack": False,
                "truth_or_source_inputs": 0,
            },
            "observation": artifact_entry(observation_path, relative_to=arm_root),
            "observation_index": artifact_entry(index_path, relative_to=arm_root),
        },
    )
    return {
        "observation": file_record(observation_path),
        "observation_index": file_record(index_path),
        "config": file_record(config_path),
    }


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan)
    validate_public_plan(plan)
    manifest, truth, truth_path = _load_private_truth(args.panel_root.resolve())
    if args.record_batch_size != 8:
        raise RuntimeError("only the frozen evaluator record batch size is accepted")
    prototype_path = args.prototype.resolve()
    table = PrototypeTable.load(
        prototype_path,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_cut_depth=CUT_DEPTH,
        expected_vocab_size=128256,
        expected_hidden_size=HIDDEN_SIZE,
    )
    del table
    prototype_digest = sha256_file(prototype_path)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    root = require_create_only_directory(args.output_root.resolve())
    public_root = root / "public"
    private_root = root / "evaluator_private"
    public_root.mkdir()
    private_root.mkdir()
    started_utc = utc_now()
    seed_everything(1701, device)
    timer_records: list[dict[str, Any]] = []
    phases_started = time.perf_counter()

    phase_started = time.perf_counter()
    public_model = load_public_model(device=device, model_path=args.public_model_path)
    public_observations = _collect_observations(
        public_model, truth, device=device, record_batch_size=args.record_batch_size
    )
    timer_records.append({"phase": "matched_public_observations", "elapsed_seconds": time.perf_counter() - phase_started})
    del public_model

    phase_started = time.perf_counter()
    target_model = load_target_model(device=device, model_path=args.target_model_path)
    shifted_observations = _collect_observations(
        target_model, truth, device=device, record_batch_size=args.record_batch_size
    )
    timer_records.append({"phase": "shifted_target_observations", "elapsed_seconds": time.perf_counter() - phase_started})
    del target_model

    arm_entries = {
        "arm-000": _write_arm(
            public_root / "arm-000",
            public_observations,
            plan=plan,
            prototype_path=prototype_path,
            prototype_digest=prototype_digest,
        ),
        "arm-001": _write_arm(
            public_root / "arm-001",
            shifted_observations,
            plan=plan,
            prototype_path=prototype_path,
            prototype_digest=prototype_digest,
        ),
    }
    # The condition map is private evaluator state.  It is intentionally not
    # copied under either public arm directory.
    map_path = private_root / "condition_map.json"
    write_json_exclusive(
        map_path,
        {
            "schema": "token-reconstruction.trr-p01-private-condition-map.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "mapping": {
                "arm-000": "matched_public",
                "arm-001": "shifted_target_lora",
            },
            "target_model": {
                "id": "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct",
                "revision": "7fa9d06a59246629244cdd3b6b92e4fc756baa0f",
                "config_sha256": TARGET_CONFIG_SHA256,
                "model_safetensors_sha256": TARGET_MODEL_SHA256,
                "model_path": str(args.target_model_path) if args.target_model_path else None,
            },
            "same_input_truth": file_record(truth_path),
        },
    )
    evidence_path = private_root / "evaluator_evidence.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": EVALUATOR_SCHEMA,
            "task_id": TASK_ID,
            "status": "PUBLIC_ARMS_FROZEN_BEFORE_SCORING",
            "truth_opened": False,
            "source_truth_included": True,
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": command_record(device),
            "environment": environment_record(device),
            "selected_device": str(device),
            "plan": file_record(args.plan),
            "panel_manifest": file_record(args.panel_root.resolve() / "panel_manifest.json"),
            "private_truth": file_record(truth_path),
            "prototype": file_record(prototype_path),
            "target_model": {
                "id": "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct",
                "revision": "7fa9d06a59246629244cdd3b6b92e4fc756baa0f",
                "config_sha256": TARGET_CONFIG_SHA256,
                "model_safetensors_sha256": TARGET_MODEL_SHA256,
            },
            "public_condition_mapping_private": file_record(map_path),
            "arms": arm_entries,
            "record_order": manifest["record_order"],
            "observation_geometry": [16, SEQUENCE_TOKENS, HIDDEN_SIZE],
            "identical_inputs": True,
            "target_weights_visible_to_attack": False,
            "condition_label_visible_to_attack": False,
            "truth_opened_by_attack": False,
            # Observation preparation runs the two full public/shifted model
            # arms.  Keep forward invocations distinct from model input-token
            # instances; this is not reconstruction-prefix simulation.
            "public_model_forward_calls": 2 * ((16 + args.record_batch_size - 1) // args.record_batch_size),
            "public_model_input_token_evaluations": 2 * 16 * SEQUENCE_TOKENS,
            "candidate_simulations": 0,
            "phases": timer_records,
            "elapsed_seconds": time.perf_counter() - phases_started,
            "peak_memory": peak_memory(device),
        },
    )
    print({"status": "PUBLIC_ARMS_FROZEN_BEFORE_SCORING", "public_root": str(public_root), "private_root": str(private_root)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
