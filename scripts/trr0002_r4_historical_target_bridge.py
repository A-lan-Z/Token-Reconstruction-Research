#!/usr/bin/env python3
"""Target-only historical-input bridge for TRR-0002 owner revision R4.

The evaluator replays the exact historical Finance token IDs and creates three
layer-4 observation tensors. Reconstruction receives only sanitized observations,
the public checkpoint, and (for the historical control) the frozen public-Alpaca
affine lens. Truth is loaded by a separate scoring command after predictions are
frozen and hash-bound.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time
from typing import Any, Mapping

from safetensors.torch import load_file, save_file
import torch
import transformers

import trr0001_r2_dual_benchmark as common
import trr0002_r2_finance_target_shortlist as owner_r2
import trr0002_r3_strict_surrogate as owner_r3
from token_reconstruction.a1a2_configuration_search import (
    decode_policy,
    resolved_policy_from_dict,
)
from token_reconstruction.component_crossover import propose_public_a1
from token_reconstruction.dual_benchmark import (
    paired_record_differences,
    score_predictions,
    scored_mask,
    validate_observations,
)
from token_reconstruction.experiment_runtime import (
    command_record,
    peak_memory,
    seed_everything,
)
from token_reconstruction.metrics import bootstrap_mean
from token_reconstruction.public_prefix import ContiguousPublicPrefix
from token_reconstruction.strict_base_surrogate import propose_checkpoint_identity


TASK_ID = "TRR-0002"
REVISION_ID = "TRR-0002-OWNER-REVISION-R4"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
BASE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
HEAVY_MODEL_ID = "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
HEAVY_REVISION = "7fa9d06a59246629244cdd3b6b92e4fc756baa0f"
FINANCE_CONFIG_SHA256 = owner_r2.SOURCE_CONFIG_SHA256
FINANCE_TRACE_SHA256 = owner_r2.SOURCE_TRACE_SHA256
HISTORICAL_LENS_SHA256 = owner_r3.HISTORICAL_LENS_SHA256
RECORD_BATCH_SIZE = 4
TARGET_GENERATION_BATCH_SIZE = 16
EXPECTED_RECORDS = 128
EXPECTED_POSITIONS = 128
EXPECTED_VALID_WITH_BOS = 14118
EXPECTED_SCORED_POST_BOS = 13990

CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "condition_id": "public_base_target_cut4",
        "short_label": "public base target",
        "target_id": BASE_MODEL_ID,
        "target_revision": BASE_REVISION,
        "weight_version": "untouched_public_checkpoint",
    },
    {
        "condition_id": "finance_generation300_target_cut4",
        "short_label": "Finance generation-300 target",
        "target_id": BASE_MODEL_ID,
        "target_revision": BASE_REVISION,
        "weight_version": "victim_post_000299",
    },
    {
        "condition_id": "vikhr_heavy_target_cut4",
        "short_label": "Vikhr heavy target",
        "target_id": HEAVY_MODEL_ID,
        "target_revision": HEAVY_REVISION,
        "weight_version": "full_supervised_fine_tune",
    },
)
CONDITION_IDS = tuple(item["condition_id"] for item in CONDITIONS)
PROPOSER_IDS = ("historical_alpaca_affine_a1", "checkpoint_identity_a1")
CHECKPOINT_POLICY_IDS = (
    "a1a2_589f6e179eb4626877c2",
    "a1a2_43ea0bb737bc075531ca",
    "a1a2_13f73c306bf8946e9a28",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
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
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(contiguous.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def require_file_record(
    expected: Mapping[str, Any], path: Path, label: str
) -> None:
    actual = file_record(path)
    for key in ("bytes", "sha256"):
        if actual[key] != expected.get(key):
            raise RuntimeError(f"{label} {key} changed after freeze")


def policies_from_r2_plan(path: Path) -> list[dict[str, Any]]:
    source = load_json(path)
    entries = list(source["shortlist_selection"]["shortlist"])
    expected = [item["policy_id"] for item in owner_r2.SHORTLIST]
    if len(entries) != 12 or [item["policy_id"] for item in entries] != expected:
        raise RuntimeError("R2 policy shortlist changed")
    for entry in entries:
        if resolved_policy_from_dict(entry["policy"]).policy_id != entry["policy_id"]:
            raise RuntimeError("R2 serialized policy identity changed")
    return entries


def expected_reproduction(path: Path) -> dict[str, dict[str, int]]:
    source = load_json(path)
    output = {
        row["policy_id"]: {
            "correct_tokens": int(row["metrics"]["correct_tokens"]),
            "scored_tokens": int(row["metrics"]["scored_tokens"]),
            "exact_records": int(row["metrics"]["exact_records"]),
        }
        for row in source["diagnostic_ranking"]
    }
    if len(output) != 12:
        raise RuntimeError("R2 reproduction registry changed")
    return output


def proposer_entries(
    plan: Mapping[str, Any], proposer_id: str
) -> list[dict[str, Any]]:
    entries = list(plan["policies"])
    if proposer_id == "historical_alpaca_affine_a1":
        return entries
    if proposer_id == "checkpoint_identity_a1":
        return [
            item for item in entries if item["policy_id"] in CHECKPOINT_POLICY_IDS
        ]
    raise RuntimeError(f"unknown proposer: {proposer_id}")


def validate_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("schema")
        != "token-reconstruction.trr0002-owner-r4-historical-target-bridge-preregistration.v1"
        or plan.get("revision_id") != REVISION_ID
        or plan.get("status") != "FROZEN_BEFORE_TARGET_OBSERVATION_GENERATION"
    ):
        raise RuntimeError("R4 plan identity changed")
    if (
        plan["benchmark"]["records"] != EXPECTED_RECORDS
        or plan["benchmark"]["scored_post_bos_tokens"]
        != EXPECTED_SCORED_POST_BOS
        or plan["execution"]["record_batch_size"] != RECORD_BATCH_SIZE
    ):
        raise RuntimeError("R4 historical benchmark constants changed")
    if [item["condition_id"] for item in plan["targets"]] != list(CONDITION_IDS):
        raise RuntimeError("R4 target order changed")
    policies = list(plan["policies"])
    if len(policies) != 12:
        raise RuntimeError("R4 policy count changed")
    for entry in policies:
        if resolved_policy_from_dict(entry["policy"]).policy_id != entry["policy_id"]:
            raise RuntimeError("R4 policy serialization changed")
    expected_cells = len(CONDITIONS) * (
        len(policies) + len(CHECKPOINT_POLICY_IDS)
    )
    if plan["matrix"]["expected_cells"] != expected_cells:
        raise RuntimeError("R4 expected matrix size changed")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command_name", required=True)

    preregister = commands.add_parser("preregister")
    preregister.add_argument("--repository-root", type=Path, default=Path("."))
    preregister.add_argument("--historical-root", type=Path, required=True)
    preregister.add_argument("--request", type=Path, required=True)
    preregister.add_argument("--r2-plan", type=Path, required=True)
    preregister.add_argument("--r2-result", type=Path, required=True)
    preregister.add_argument("--r3-plan", type=Path, required=True)
    preregister.add_argument("--base-model-path", type=Path, required=True)
    preregister.add_argument("--heavy-model-path", type=Path, required=True)
    preregister.add_argument("--historical-lens", type=Path, required=True)
    preregister.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repository-root", type=Path, default=Path("."))
    prepare.add_argument("--historical-root", type=Path, required=True)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--base-model-path", type=Path, required=True)
    prepare.add_argument("--heavy-model-path", type=Path, required=True)
    prepare.add_argument("--input-root", type=Path, required=True)
    prepare.add_argument("--truth-sidecar", type=Path, required=True)
    prepare.add_argument("--evidence", type=Path, required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--repository-root", type=Path, default=Path("."))
    predict.add_argument("--plan", type=Path, required=True)
    predict.add_argument("--input-root", type=Path, required=True)
    predict.add_argument("--model-path", type=Path, required=True)
    predict.add_argument("--proposer", choices=PROPOSER_IDS, required=True)
    predict.add_argument("--condition-id", choices=CONDITION_IDS)
    predict.add_argument("--policy-id")
    predict.add_argument("--lens-path", type=Path)
    predict.add_argument("--output-directory", type=Path, required=True)

    combine = commands.add_parser("combine")
    combine.add_argument("--repository-root", type=Path, default=Path("."))
    combine.add_argument("--plan", type=Path, required=True)
    combine.add_argument("--input-root", type=Path, required=True)
    combine.add_argument(
        "--part-directory", action="append", type=Path, required=True
    )
    combine.add_argument("--output-directory", type=Path, required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--repository-root", type=Path, default=Path("."))
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--input-root", type=Path, required=True)
    freeze.add_argument(
        "--prediction-directory", action="append", type=Path, required=True
    )
    freeze.add_argument("--output", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--repository-root", type=Path, default=Path("."))
    score.add_argument("--historical-root", type=Path, required=True)
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--freeze-receipt", type=Path, required=True)
    score.add_argument("--truth-sidecar", type=Path, required=True)
    score.add_argument("--preparation-evidence", type=Path, required=True)
    score.add_argument(
        "--prediction-directory", action="append", type=Path, required=True
    )
    score.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--preparation-evidence", type=Path, required=True)
    validate.add_argument("--freeze-receipt", type=Path, required=True)
    validate.add_argument("--result", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    return root


def command_preregister(args: argparse.Namespace) -> int:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("R4 preregistration is create-only")
    root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    config_path = historical_root / owner_r2.SOURCE_CONFIG_RELATIVE
    config = load_json(config_path)
    trace_path = historical_root / config["source"]["path"]
    if sha256_file(config_path) != FINANCE_CONFIG_SHA256:
        raise RuntimeError("historical Finance config changed")
    if sha256_file(trace_path) != FINANCE_TRACE_SHA256:
        raise RuntimeError("historical Finance trace changed")
    if sha256_file(args.historical_lens) != HISTORICAL_LENS_SHA256:
        raise RuntimeError("historical Alpaca lens changed")
    entries = policies_from_r2_plan(args.r2_plan)
    reproduction = expected_reproduction(args.r2_result)
    base_path = args.base_model_path.resolve(strict=True)
    heavy_path = args.heavy_model_path.resolve(strict=True)
    payload = {
        "schema": "token-reconstruction.trr0002-owner-r4-historical-target-bridge-preregistration.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "FROZEN_BEFORE_TARGET_OBSERVATION_GENERATION",
        "created_utc": utc_now(),
        "created_from_commit": git_head(root),
        "owner_request": file_record(args.request),
        "scientific_question": (
            "With the exact historical 128-record Finance token matrix fixed, "
            "how does A1+A2 reconstruction change when only the unavailable "
            "target prefix weights change?"
        ),
        "benchmark": {
            "dataset": config["truth"]["dataset"],
            "split": config["truth"]["split"],
            "fingerprint": config["truth"]["fingerprint"],
            "shuffle_seed": config["truth"]["shuffle_seed"],
            "raw_cursor_start": config["truth"]["raw_cursor_start"],
            "raw_cursor_end_exclusive": config["truth"]["raw_cursor_end_exclusive"],
            "historical_chat_template_date": config["truth"][
                "historical_chat_template_date"
            ],
            "records": EXPECTED_RECORDS,
            "positions": EXPECTED_POSITIONS,
            "valid_tokens_including_bos": EXPECTED_VALID_WITH_BOS,
            "scored_post_bos_tokens": EXPECTED_SCORED_POST_BOS,
            "cut_depth": 4,
            "metric": (
                "post-BOS token accuracy; abstentions and unattempted suffix "
                "tokens are wrong"
            ),
            "exact_record_metric": (
                "all post-BOS token IDs in the record are correct"
            ),
            "source_config": file_record(config_path),
            "original_finance_activation_trace": file_record(trace_path),
        },
        "invariants": {
            "same_exact_token_ids": True,
            "same_attention_masks": True,
            "same_position_ids": True,
            "same_record_order": True,
            "same_cut_depth": True,
            "same_A2_public_prefix": True,
            "same_policy_constants": True,
            "only_target_prefix_weights_change_across_conditions": True,
        },
        "targets": [
            {
                **CONDITIONS[0],
                "model_artifacts": {
                    name: file_record(base_path / name)
                    for name in (
                        "config.json",
                        "model.safetensors",
                        "tokenizer.json",
                    )
                },
            },
            {
                **CONDITIONS[1],
                "fine_tuning_dataset": config["truth"]["dataset"],
                "checkpoint_generation": config["source"][
                    "checkpoint_generation"
                ],
                "target_step": config["source"]["target_step"],
                "activation_source": file_record(trace_path),
            },
            {
                **CONDITIONS[2],
                "fine_tuning_dataset": owner_r3.HEAVY_DATASET_ID,
                "model_artifacts": {
                    name: file_record(heavy_path / name)
                    for name in (
                        "config.json",
                        "model.safetensors",
                        "tokenizer.json",
                    )
                },
            },
        ],
        "reconstruction_resources": {
            "a2": {
                "model_id": BASE_MODEL_ID,
                "revision": BASE_REVISION,
                "layers": [0, 1, 2, 3],
                "target_weights_or_calls_available": False,
            },
            "proposers": [
                {
                    "proposer_id": "historical_alpaca_affine_a1",
                    "label": (
                        "historical A1 (public-Alpaca-fitted affine lens)"
                    ),
                    "lens": file_record(args.historical_lens),
                    "fitted_parameters": 4_196_353,
                    "auxiliary_training_rows": 52_002,
                    "target_information_used": False,
                    "policy_scope": "all 12 frozen R2 policies",
                },
                {
                    "proposer_id": "checkpoint_identity_a1",
                    "label": "plain untouched public-checkpoint A1 control",
                    "lens": None,
                    "fitted_parameters": 0,
                    "auxiliary_training_rows": 0,
                    "target_information_used": False,
                    "policy_scope": (
                        "fixed K64 direct, K256 direct, and K512 centered"
                    ),
                },
            ],
        },
        "policies": entries,
        "matrix": {
            "historical_alpaca_cells": len(CONDITIONS) * len(entries),
            "checkpoint_identity_cells": (
                len(CONDITIONS) * len(CHECKPOINT_POLICY_IDS)
            ),
            "expected_cells": len(CONDITIONS)
            * (len(entries) + len(CHECKPOINT_POLICY_IDS)),
            "report_every_cell": True,
            "no_post_score_method_selection": True,
        },
        "execution": {
            "target_generation_batch_size": TARGET_GENERATION_BATCH_SIZE,
            "record_batch_size": RECORD_BATCH_SIZE,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "candidate_budget_maximum": 512,
            "prediction_process_truth_argument": None,
            "prediction_process_dataset_inputs": 0,
            "prediction_process_target_prefix_calls": 0,
            "freeze_before_scoring": True,
        },
        "reproduction_gate": {
            "reference": file_record(args.r2_result),
            "required_condition": "finance_generation300_target_cut4",
            "required_proposer": "historical_alpaca_affine_a1",
            "required_exact_metric_match_for_all_12_policies": True,
            "expected": reproduction,
        },
        "prior_evidence": {
            "r2_plan": file_record(args.r2_plan),
            "r3_plan": file_record(args.r3_plan),
            "r3_grandmaster_results_are_auxiliary_not_target_only": True,
        },
        "reporting": {
            "primary": [
                "correct post-BOS tokens",
                "token accuracy",
                "exact records",
            ],
            "secondary": [
                "candidate recall",
                "candidate simulations",
                "runtime",
                "peak memory",
            ],
            "comparisons": (
                "within-policy target-only deltas relative to Finance generation 300"
            ),
        },
        "claim_limits": [
            (
                "historical Finance truth was open before R4, so this is a "
                "retrospective diagnostic"
            ),
            (
                "the GrandMaster-input R3 panel is auxiliary and is not "
                "numerically pooled with this bridge"
            ),
            (
                "the checkpoint-identity proposer is a separate control, "
                "not the historical A1"
            ),
            "no unseen-data generalization claim follows from this bridge",
        ],
    }
    validate_plan(payload)
    write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cells": payload["matrix"]["expected_cells"],
            },
            sort_keys=True,
        )
    )
    return 0


@torch.inference_mode()
def generate_historical_observations(
    model_path: Path,
    truth: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float, dict[str, int]]:
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.requires_grad_(False)
    if (
        int(model.config.hidden_size) != 2048
        or int(model.config.vocab_size) != 128256
        or len(model.model.layers) != 16
    ):
        raise RuntimeError("target model architecture changed")
    prefix = ContiguousPublicPrefix(model, cut_depth=4).to(device).eval()
    output = torch.empty(
        (EXPECTED_RECORDS, EXPECTED_POSITIONS, 2048), dtype=torch.bfloat16
    )
    torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, EXPECTED_RECORDS, TARGET_GENERATION_BATCH_SIZE):
        stop = min(
            start + TARGET_GENERATION_BATCH_SIZE, EXPECTED_RECORDS
        )
        ids = truth[start:stop].to(device=device, dtype=torch.long)
        hidden = prefix.forward_full(ids)
        output[start:stop] = hidden.detach().cpu()
        del ids, hidden
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    memory = {
        "peak_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_cuda_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
    }
    del prefix, model
    gc.collect()
    torch.cuda.empty_cache()
    return output.contiguous(), elapsed, memory


def command_prepare(args: argparse.Namespace) -> int:
    outputs = (
        args.input_root / "config.json",
        args.input_root / "observations.safetensors",
        args.truth_sidecar,
        args.evidence,
    )
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise RuntimeError("R4 preparation outputs are create-only")
    root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    plan = load_json(args.plan)
    validate_plan(plan)
    config_path = historical_root / owner_r2.SOURCE_CONFIG_RELATIVE
    trace_path = historical_root / load_json(config_path)["source"]["path"]
    require_file_record(
        plan["benchmark"]["source_config"], config_path, "Finance config"
    )
    require_file_record(
        plan["benchmark"]["original_finance_activation_trace"],
        trace_path,
        "Finance trace",
    )

    started_utc = utc_now()
    source_path = (
        historical_root / "scripts/score_a1_a2_source300_20260809.py"
    )
    source300 = common.import_path(
        "trr0002_r4_source300_prepare", source_path
    )
    config, captures, finance, mask, positions = common.historical_inputs(
        historical_root, source300
    )
    truth, record_ids = common.load_old_truth(
        source300, captures, config
    )
    valid_with_bos = int(mask.sum().item())
    scored = int(scored_mask(mask).sum().item())
    if (
        valid_with_bos != EXPECTED_VALID_WITH_BOS
        or scored != EXPECTED_SCORED_POST_BOS
    ):
        raise RuntimeError("historical Finance token counts changed")
    expected_positions = mask.cumsum(dim=-1).sub(1).clamp_min(0)
    if not torch.equal(positions, expected_positions):
        raise RuntimeError("historical position IDs changed")

    if not torch.cuda.is_available():
        raise RuntimeError("R4 target preparation requires CUDA")
    device = torch.device("cuda")
    seed_everything(20260904)
    base, base_seconds, base_memory = generate_historical_observations(
        args.base_model_path.resolve(strict=True), truth, device=device
    )
    heavy, heavy_seconds, heavy_memory = generate_historical_observations(
        args.heavy_model_path.resolve(strict=True), truth, device=device
    )
    for observations in (base, finance, heavy):
        validate_observations(observations, mask, positions)

    args.input_root.mkdir(parents=True, exist_ok=False)
    observation_path = args.input_root / "observations.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    for condition_id, observations in zip(
        CONDITION_IDS, (base, finance, heavy), strict=True
    ):
        tensors[
            f"{condition_id}.activations"
        ] = observations.contiguous()
        tensors[
            f"{condition_id}.attention_mask"
        ] = mask.to(torch.uint8).contiguous()
        tensors[
            f"{condition_id}.position_ids"
        ] = positions.to(torch.int32).contiguous()
    save_file(
        tensors,
        observation_path,
        metadata={
            "schema": (
                "token-reconstruction.trr0002-owner-r4-"
                "sanitized-observations.v1"
            ),
            "truth_included": "false",
            "source_text_included": "false",
            "records": str(EXPECTED_RECORDS),
            "positions": str(EXPECTED_POSITIONS),
        },
    )
    sanitized = {
        "schema": (
            "token-reconstruction.trr0002-owner-r4-sanitized-input.v1"
        ),
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "plan_sha256": sha256_file(args.plan),
        "truth_included": False,
        "dataset_inputs_included": False,
        "source_text_or_hash_included": False,
        "records": EXPECTED_RECORDS,
        "positions": EXPECTED_POSITIONS,
        "hidden_size": 2048,
        "conditions": list(CONDITIONS),
        "opaque_record_ids": [
            f"historical-finance-{index:06d}"
            for index in range(1, EXPECTED_RECORDS + 1)
        ],
        "policies": plan["policies"],
        "record_batch_size": RECORD_BATCH_SIZE,
        "permitted_files": ["config.json", "observations.safetensors"],
    }
    write_json_exclusive(args.input_root / "config.json", sanitized)

    args.truth_sidecar.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "truth_token_ids": truth.to(torch.int32).contiguous(),
            "attention_mask": mask.to(torch.uint8).contiguous(),
            "position_ids": positions.to(torch.int32).contiguous(),
        },
        args.truth_sidecar,
        metadata={
            "schema": (
                "token-reconstruction.trr0002-owner-r4-evaluator-truth.v1"
            ),
            "access": "evaluator-only-after-freeze",
        },
    )
    evidence = {
        "schema": (
            "token-reconstruction.trr0002-owner-r4-"
            "preparation-evidence.v1"
        ),
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": (
            "EXACT_HISTORICAL_INPUTS_AND_THREE_TARGET_OBSERVATIONS_PREPARED"
        ),
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "execution_commit": git_head(root),
        "command": command_record(),
        "exit_status": 0,
        "plan": file_record(args.plan),
        "sanitized_config": file_record(
            args.input_root / "config.json"
        ),
        "sanitized_observations": file_record(observation_path),
        "evaluator_truth_sidecar": file_record(args.truth_sidecar),
        "exact_input_identity": {
            "truth_token_ids_sha256": tensor_sha256(truth),
            "attention_mask_sha256": tensor_sha256(mask),
            "position_ids_sha256": tensor_sha256(positions),
            "record_ids_sha256": hashlib.sha256(
                "\n".join(record_ids).encode("utf-8")
            ).hexdigest(),
            "records": int(truth.shape[0]),
            "positions": int(truth.shape[1]),
            "valid_tokens_including_bos": valid_with_bos,
            "scored_post_bos_tokens": scored,
        },
        "target_generation": {
            "public_base_target_cut4": {
                "seconds": base_seconds,
                "activation_sha256": tensor_sha256(base),
                **base_memory,
            },
            "finance_generation300_target_cut4": {
                "seconds": 0.0,
                "activation_sha256": tensor_sha256(finance),
                "source": (
                    "exact retained victim_post_000299 activation trace"
                ),
            },
            "vikhr_heavy_target_cut4": {
                "seconds": heavy_seconds,
                "activation_sha256": tensor_sha256(heavy),
                **heavy_memory,
            },
            "batch_size": TARGET_GENERATION_BATCH_SIZE,
            "dtype": "bfloat16",
            "cut_depth": 4,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "process_max_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "peak_memory": peak_memory(),
        },
    }
    write_json_exclusive(args.evidence, evidence)
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "truth_sha256": evidence["exact_input_identity"][
                    "truth_token_ids_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def load_conditions(
    input_root: Path, config: Mapping[str, Any]
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    state = load_file(
        input_root / "observations.safetensors", device="cpu"
    )
    output = {}
    expected: set[str] = set()
    for item in config["conditions"]:
        condition_id = item["condition_id"]
        keys = (
            f"{condition_id}.activations",
            f"{condition_id}.attention_mask",
            f"{condition_id}.position_ids",
        )
        expected.update(keys)
        observations = state[keys[0]].contiguous()
        mask = state[keys[1]].long().contiguous()
        positions = state[keys[2]].long().contiguous()
        if observations.shape != (
            EXPECTED_RECORDS,
            EXPECTED_POSITIONS,
            2048,
        ):
            raise RuntimeError("R4 observation geometry changed")
        validate_observations(observations, mask, positions)
        output[condition_id] = (observations, mask, positions)
    if set(state) != expected:
        raise RuntimeError(
            "R4 sanitized observation tensor registry changed"
        )
    return output


def tensors_equal_with_matching_nan(
    left: torch.Tensor, right: torch.Tensor
) -> bool:
    """Require exact values while allowing NaN at the same positions."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if torch.equal(left, right):
        return True
    if not (left.is_floating_point() or left.is_complex()):
        return False
    left_nan = torch.isnan(left)
    right_nan = torch.isnan(right)
    return torch.equal(left_nan, right_nan) and torch.equal(
        left[~left_nan], right[~right_nan]
    )


def command_predict(args: argparse.Namespace) -> int:
    if (
        args.output_directory.exists()
        or args.output_directory.is_symlink()
    ):
        raise RuntimeError("R4 prediction output is create-only")
    root = args.repository_root.resolve(strict=True)
    plan = load_json(args.plan)
    validate_plan(plan)
    config = load_json(args.input_root / "config.json")
    if (
        config.get("schema")
        != "token-reconstruction.trr0002-owner-r4-sanitized-input.v1"
        or config.get("truth_included") is not False
        or config.get("dataset_inputs_included") is not False
        or config.get("source_text_or_hash_included") is not False
        or config.get("plan_sha256") != sha256_file(args.plan)
    ):
        raise RuntimeError("R4 sanitized input config is invalid")
    if args.proposer == "historical_alpaca_affine_a1":
        if (
            args.lens_path is None
            or sha256_file(args.lens_path) != HISTORICAL_LENS_SHA256
        ):
            raise RuntimeError(
                "historical A1 requires the exact frozen Alpaca lens"
            )
    elif args.lens_path is not None:
        raise RuntimeError(
            "checkpoint-identity A1 may not receive a fitted lens"
        )

    entries = proposer_entries(plan, args.proposer)
    observations_by_condition = load_conditions(
        args.input_root, config
    )
    if args.condition_id is not None:
        if args.condition_id not in observations_by_condition:
            raise RuntimeError("R4 condition filter is absent")
        observations_by_condition = {
            args.condition_id: observations_by_condition[args.condition_id]
        }
    if args.policy_id is not None:
        entries = [
            entry
            for entry in entries
            if entry["policy_id"] == args.policy_id
        ]
        if len(entries) != 1:
            raise RuntimeError(
                "R4 policy filter is absent for this proposer"
            )
    prefix, embeddings, device = owner_r3.load_public_surrogate(
        args.model_path.resolve(strict=True)
    )
    lens = None
    if args.lens_path is not None:
        reference = common.import_path(
            "trr0002_r4_reference",
            root / "reference/strict_bos/round001_teacher.py",
        )
        lens = reference.load_frozen_lens(
            args.lens_path, device=device
        )
    seed_everything(20260904)
    started_utc = utc_now()
    tensors: dict[str, torch.Tensor] = {}
    costs: dict[str, Any] = {}
    for condition_id, values in observations_by_condition.items():
        observations, mask, positions = values
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if args.proposer == "checkpoint_identity_a1":
            proposal = propose_checkpoint_identity(
                observations=observations,
                attention_mask=mask,
                normalized_embeddings=embeddings,
            )
        else:
            assert lens is not None
            proposal = propose_public_a1(
                observations=observations,
                attention_mask=mask,
                lens=lens,
                normalized_embeddings=embeddings,
            )
        tensors[
            f"{condition_id}.candidates_top512"
        ] = proposal.candidates.to(torch.int32)
        tensors[
            f"{condition_id}.a1_confidence"
        ] = proposal.top1_confidence.float()
        condition_costs: dict[str, Any] = {}
        for index, entry in enumerate(entries, start=1):
            policy = resolved_policy_from_dict(entry["policy"])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            result = decode_policy(
                observations=observations,
                attention_mask=mask,
                position_ids=positions,
                candidates=proposal.candidates,
                a1_confidence=proposal.top1_confidence,
                precut=prefix,
                device=device,
                policy=policy,
                record_batch_size=RECORD_BATCH_SIZE,
            )
            key = f"{condition_id}.{policy.policy_id}"
            tensors[
                f"{key}.predictions"
            ] = result.predictions.to(torch.int32)
            tensors[
                f"{key}.routes"
            ] = result.routes.to(torch.int8)
            tensors[
                f"{key}.selected_k"
            ] = result.selected_k.to(torch.int16)
            tensors[
                f"{key}.selected_signal"
            ] = result.selected_signal.float()
            condition_costs[policy.policy_id] = {
                "label": entry["label"],
                "proposal_seconds_shared": proposal.elapsed_seconds,
                "selection_seconds": result.elapsed_seconds,
                "method_compute_seconds": (
                    proposal.elapsed_seconds + result.elapsed_seconds
                ),
                "candidate_simulations": result.candidate_simulations,
                "executed_candidate_simulations": (
                    result.executed_candidate_simulations
                ),
                "prefix_commit_tokens": result.prefix_commit_tokens,
                "record_batch_size": result.record_batch_size,
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_cuda_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
            print(
                json.dumps(
                    {
                        "status": "R4_PREDICTION_PROGRESS",
                        "proposer": args.proposer,
                        "condition": condition_id,
                        "completed": index,
                        "total": len(entries),
                        "policy": entry["label"],
                        "seconds": result.elapsed_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        costs[condition_id] = {
            "proposal": {
                "seconds": proposal.elapsed_seconds,
                "candidate_budget": 512,
                "scored_positions": int(
                    scored_mask(mask).sum().item()
                ),
            },
            "policies": condition_costs,
        }

    args.output_directory.mkdir(parents=True, exist_ok=False)
    prediction_path = (
        args.output_directory / "predictions.safetensors"
    )
    save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        prediction_path,
        metadata={
            "schema": (
                "token-reconstruction.trr0002-owner-r4-predictions.v1"
            ),
            "task_id": TASK_ID,
            "revision_id": REVISION_ID,
            "proposer_id": args.proposer,
            "truth_loaded": "false",
            "target_prefix_calls": "0",
        },
    )
    evidence = {
        "schema": (
            "token-reconstruction.trr0002-owner-r4-"
            "reconstruction-evidence.v1"
        ),
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "PREDICTIONS_CREATED_WITHOUT_TRUTH",
        "proposer_id": args.proposer,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "execution_commit": git_head(root),
        "command": command_record(),
        "exit_status": 0,
        "plan": file_record(args.plan),
        "sanitized_config": file_record(
            args.input_root / "config.json"
        ),
        "sanitized_observations": file_record(
            args.input_root / "observations.safetensors"
        ),
        "predictions": file_record(prediction_path),
        "access": {
            "truth_arguments": 0,
            "dataset_arguments": 0,
            "target_weights_available": False,
            "target_prefix_calls": 0,
            "public_surrogate_only": True,
            "fitted_lens_available": args.lens_path is not None,
        },
        "policy_count": len(entries),
        "policy_ids": [entry["policy_id"] for entry in entries],
        "condition_count": len(observations_by_condition),
        "condition_ids": list(observations_by_condition),
        "costs": costs,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "process_max_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "peak_memory": peak_memory(),
            "pid": os.getpid(),
        },
    }
    write_json_exclusive(
        args.output_directory / "evidence.json", evidence
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "proposer": args.proposer,
                "sha256": evidence["predictions"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_combine(args: argparse.Namespace) -> int:
    if (
        args.output_directory.exists()
        or args.output_directory.is_symlink()
    ):
        raise RuntimeError("R4 combined output is create-only")
    root = args.repository_root.resolve(strict=True)
    plan = load_json(args.plan)
    validate_plan(plan)
    if not args.part_directory:
        raise RuntimeError("R4 combine received no parts")

    proposer_id: str | None = None
    tensors: dict[str, torch.Tensor] = {}
    costs = {
        condition_id: {"proposal": None, "policies": {}}
        for condition_id in CONDITION_IDS
    }
    parts: list[dict[str, Any]] = []
    observed_cells: set[tuple[str, str]] = set()
    for directory in args.part_directory:
        evidence_path = directory / "evidence.json"
        prediction_path = directory / "predictions.safetensors"
        evidence = load_json(evidence_path)
        current_proposer = evidence.get("proposer_id")
        if proposer_id is None:
            proposer_id = current_proposer
        if current_proposer != proposer_id:
            raise RuntimeError("R4 combine mixed proposers")
        if (
            evidence.get("status")
            != "PREDICTIONS_CREATED_WITHOUT_TRUTH"
            or evidence.get("condition_count") != 1
            or evidence.get("policy_count") != 1
            or evidence["access"]["truth_arguments"] != 0
            or evidence["access"]["target_prefix_calls"] != 0
        ):
            raise RuntimeError("R4 prediction part is invalid")
        require_file_record(
            evidence["plan"], args.plan, "part plan"
        )
        require_file_record(
            evidence["predictions"],
            prediction_path,
            "part prediction",
        )
        condition_id = evidence["condition_ids"][0]
        policy_id = evidence["policy_ids"][0]
        if (condition_id, policy_id) in observed_cells:
            raise RuntimeError("duplicate R4 prediction cell")
        observed_cells.add((condition_id, policy_id))
        state = load_file(prediction_path, device="cpu")
        candidate_key = f"{condition_id}.candidates_top512"
        confidence_key = f"{condition_id}.a1_confidence"
        expected_keys = {candidate_key, confidence_key}
        cell_prefix = f"{condition_id}.{policy_id}"
        expected_keys.update(
            f"{cell_prefix}.{suffix}"
            for suffix in (
                "predictions",
                "routes",
                "selected_k",
                "selected_signal",
            )
        )
        if set(state) != expected_keys:
            raise RuntimeError(
                "R4 prediction part tensor registry changed"
            )
        for key in (candidate_key, confidence_key):
            if key in tensors:
                if not tensors_equal_with_matching_nan(
                    tensors[key], state[key]
                ):
                    raise RuntimeError(
                        "R4 repeated proposal tensors differ"
                    )
            else:
                tensors[key] = state[key].contiguous()
        for suffix in (
            "predictions",
            "routes",
            "selected_k",
            "selected_signal",
        ):
            key = f"{cell_prefix}.{suffix}"
            tensors[key] = state[key].contiguous()
        if costs[condition_id]["proposal"] is None:
            costs[condition_id]["proposal"] = evidence["costs"][
                condition_id
            ]["proposal"]
        costs[condition_id]["policies"][policy_id] = evidence[
            "costs"
        ][condition_id]["policies"][policy_id]
        parts.append(
            {
                "directory": str(directory),
                "condition_id": condition_id,
                "policy_id": policy_id,
                "predictions": file_record(prediction_path),
                "evidence": file_record(evidence_path),
            }
        )

    if proposer_id not in PROPOSER_IDS:
        raise RuntimeError("R4 combined proposer is invalid")
    entries = proposer_entries(plan, proposer_id)
    expected_cells = {
        (condition_id, entry["policy_id"])
        for condition_id in CONDITION_IDS
        for entry in entries
    }
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise RuntimeError(
            f"R4 prediction parts incomplete: missing={missing}, extra={extra}"
        )
    if any(value["proposal"] is None for value in costs.values()):
        raise RuntimeError("R4 combined proposal costs are incomplete")

    args.output_directory.mkdir(parents=True, exist_ok=False)
    prediction_path = (
        args.output_directory / "predictions.safetensors"
    )
    save_file(
        tensors,
        prediction_path,
        metadata={
            "schema": (
                "token-reconstruction.trr0002-owner-r4-"
                "combined-predictions.v1"
            ),
            "task_id": TASK_ID,
            "revision_id": REVISION_ID,
            "proposer_id": proposer_id,
            "truth_loaded": "false",
            "target_prefix_calls": "0",
            "part_count": str(len(parts)),
        },
    )
    evidence = {
        "schema": (
            "token-reconstruction.trr0002-owner-r4-"
            "combined-reconstruction-evidence.v1"
        ),
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "PREDICTIONS_CREATED_WITHOUT_TRUTH",
        "proposer_id": proposer_id,
        "created_utc": utc_now(),
        "execution_commit": git_head(root),
        "command": command_record(),
        "exit_status": 0,
        "plan": file_record(args.plan),
        "sanitized_config": file_record(
            args.input_root / "config.json"
        ),
        "sanitized_observations": file_record(
            args.input_root / "observations.safetensors"
        ),
        "predictions": file_record(prediction_path),
        "access": {
            "truth_arguments": 0,
            "dataset_arguments": 0,
            "target_weights_available": False,
            "target_prefix_calls": 0,
            "public_surrogate_only": True,
            "fitted_lens_available": (
                proposer_id == "historical_alpaca_affine_a1"
            ),
        },
        "policy_count": len(entries),
        "policy_ids": [entry["policy_id"] for entry in entries],
        "condition_count": len(CONDITIONS),
        "condition_ids": list(CONDITION_IDS),
        "costs": costs,
        "parts": parts,
    }
    write_json_exclusive(
        args.output_directory / "evidence.json", evidence
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "proposer": proposer_id,
                "parts": len(parts),
                "sha256": evidence["predictions"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_freeze(args: argparse.Namespace) -> int:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("R4 freeze receipt is create-only")
    plan = load_json(args.plan)
    validate_plan(plan)
    if len(args.prediction_directory) != len(PROPOSER_IDS):
        raise RuntimeError(
            "R4 freeze requires exactly two proposer outputs"
        )
    frozen: dict[str, Any] = {}
    for directory in args.prediction_directory:
        evidence_path = directory / "evidence.json"
        prediction_path = directory / "predictions.safetensors"
        evidence = load_json(evidence_path)
        proposer_id = evidence.get("proposer_id")
        if proposer_id not in PROPOSER_IDS or proposer_id in frozen:
            raise RuntimeError(
                "R4 proposer freeze identity is invalid"
            )
        if (
            evidence.get("status")
            != "PREDICTIONS_CREATED_WITHOUT_TRUTH"
            or evidence["access"]["truth_arguments"] != 0
            or evidence["access"]["target_prefix_calls"] != 0
        ):
            raise RuntimeError(
                "R4 reconstruction access evidence failed"
            )
        require_file_record(
            evidence["predictions"],
            prediction_path,
            "prediction",
        )
        frozen[proposer_id] = {
            "directory": str(directory),
            "predictions": file_record(prediction_path),
            "evidence": file_record(evidence_path),
        }
    if set(frozen) != set(PROPOSER_IDS):
        raise RuntimeError("R4 proposer outputs are incomplete")
    receipt = {
        "schema": (
            "token-reconstruction.trr0002-owner-r4-freeze-receipt.v1"
        ),
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "ALL_45_CELLS_FROZEN_BEFORE_SCORING",
        "created_utc": utc_now(),
        "execution_commit": git_head(
            args.repository_root.resolve(strict=True)
        ),
        "command": command_record(),
        "plan": file_record(args.plan),
        "sanitized_config": file_record(
            args.input_root / "config.json"
        ),
        "sanitized_observations": file_record(
            args.input_root / "observations.safetensors"
        ),
        "proposers": frozen,
        "expected_cells": plan["matrix"]["expected_cells"],
        "truth_loaded_by_prediction_processes": False,
        "target_prefix_calls_by_prediction_processes": 0,
    }
    write_json_exclusive(args.output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "proposers": sorted(frozen),
            },
            sort_keys=True,
        )
    )
    return 0


def count_values(
    values: torch.Tensor, mask: torch.Tensor
) -> dict[str, int]:
    selected = values[mask]
    unique, frequency = torch.unique(
        selected, return_counts=True
    )
    return {
        str(int(key)): int(count)
        for key, count in zip(unique, frequency, strict=True)
    }


def candidate_recall(
    candidates: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> dict[str, Any]:
    selected = scored_mask(mask)
    expected = truth[selected].long()
    hits = (
        candidates[selected, :k]
        .long()
        .eq(expected[:, None])
        .any(dim=1)
    )
    return {
        "k": k,
        "hits": int(hits.sum().item()),
        "scored_tokens": int(hits.numel()),
        "recall": int(hits.sum().item()) / int(hits.numel()),
    }


def command_score(args: argparse.Namespace) -> int:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("R4 score output is create-only")
    root = args.repository_root.resolve(strict=True)
    plan = load_json(args.plan)
    validate_plan(plan)
    receipt = load_json(args.freeze_receipt)
    if (
        receipt.get("status")
        != "ALL_45_CELLS_FROZEN_BEFORE_SCORING"
        or receipt.get("truth_loaded_by_prediction_processes")
        is not False
        or receipt.get(
            "target_prefix_calls_by_prediction_processes"
        )
        != 0
        or receipt.get("expected_cells")
        != plan["matrix"]["expected_cells"]
    ):
        raise RuntimeError("R4 freeze receipt is invalid")
    require_file_record(receipt["plan"], args.plan, "plan")
    preparation = load_json(args.preparation_evidence)
    if (
        preparation.get("status")
        != "EXACT_HISTORICAL_INPUTS_AND_THREE_TARGET_OBSERVATIONS_PREPARED"
    ):
        raise RuntimeError("R4 preparation evidence is invalid")
    require_file_record(
        preparation["evaluator_truth_sidecar"],
        args.truth_sidecar,
        "truth sidecar",
    )
    truth_state = load_file(args.truth_sidecar, device="cpu")
    truth = truth_state["truth_token_ids"].long().contiguous()
    mask = truth_state["attention_mask"].long().contiguous()
    positions = (
        truth_state["position_ids"].long().contiguous()
    )
    if (
        tensor_sha256(truth)
        != preparation["exact_input_identity"][
            "truth_token_ids_sha256"
        ]
        or tensor_sha256(mask)
        != preparation["exact_input_identity"][
            "attention_mask_sha256"
        ]
        or tensor_sha256(positions)
        != preparation["exact_input_identity"][
            "position_ids_sha256"
        ]
    ):
        raise RuntimeError(
            "R4 exact historical input identity changed"
        )
    record_ids = [
        f"source300-row-{index:03d}"
        for index in range(EXPECTED_RECORDS)
    ]
    selected = scored_mask(mask)
    started_utc = utc_now()

    directories: dict[str, Path] = {}
    for directory in args.prediction_directory:
        evidence = load_json(directory / "evidence.json")
        proposer_id = evidence["proposer_id"]
        if proposer_id in directories:
            raise RuntimeError(
                "duplicate R4 proposer score input"
            )
        directories[proposer_id] = directory
        frozen = receipt["proposers"][proposer_id]
        require_file_record(
            frozen["predictions"],
            directory / "predictions.safetensors",
            "prediction",
        )
        require_file_record(
            frozen["evidence"],
            directory / "evidence.json",
            "reconstruction evidence",
        )
    if set(directories) != set(PROPOSER_IDS):
        raise RuntimeError("R4 score inputs are incomplete")

    cells: list[dict[str, Any]] = []
    per_record_lookup: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = {}
    candidate_curves: list[dict[str, Any]] = []
    for proposer_id, directory in directories.items():
        evidence = load_json(directory / "evidence.json")
        frozen = load_file(
            directory / "predictions.safetensors",
            device="cpu",
        )
        entries = proposer_entries(plan, proposer_id)
        for condition_id in CONDITION_IDS:
            candidates = frozen[
                f"{condition_id}.candidates_top512"
            ].long()
            for k in (32, 64, 128, 256, 512):
                candidate_curves.append(
                    {
                        "proposer_id": proposer_id,
                        "condition_id": condition_id,
                        **candidate_recall(
                            candidates, truth, mask, k
                        ),
                    }
                )
            for entry in entries:
                policy_id = entry["policy_id"]
                key = f"{condition_id}.{policy_id}"
                predictions = frozen[
                    f"{key}.predictions"
                ].long()
                routes = frozen[
                    f"{key}.routes"
                ].to(torch.int8)
                selected_k = frozen[
                    f"{key}.selected_k"
                ].to(torch.int16)
                maximum_k = max(
                    1, int(selected_k[selected].max().item())
                )
                metrics, per_record = score_predictions(
                    predictions=predictions,
                    truth=truth,
                    attention_mask=mask,
                    candidates=candidates[:, :, :maximum_k],
                    record_ids=record_ids,
                )
                per_record_lookup[
                    (proposer_id, condition_id, policy_id)
                ] = per_record
                cells.append(
                    {
                        "proposer_id": proposer_id,
                        "condition_id": condition_id,
                        "label": entry["label"],
                        "policy_id": policy_id,
                        "policy": entry["policy"],
                        "metrics": metrics,
                        "per_record": per_record,
                        "routes": count_values(routes, selected),
                        "selected_k": count_values(
                            selected_k, selected
                        ),
                        "maximum_executed_k": maximum_k,
                        "cost": evidence["costs"][
                            condition_id
                        ]["policies"][policy_id],
                    }
                )

    if len(cells) != plan["matrix"]["expected_cells"]:
        raise RuntimeError("R4 scored matrix is incomplete")
    reproduction_rows: list[dict[str, Any]] = []
    for cell in cells:
        if (
            cell["proposer_id"]
            == plan["reproduction_gate"][
                "required_proposer"
            ]
            and cell["condition_id"]
            == plan["reproduction_gate"][
                "required_condition"
            ]
        ):
            expected = plan["reproduction_gate"]["expected"][
                cell["policy_id"]
            ]
            actual = {
                "correct_tokens": cell["metrics"][
                    "correct_tokens"
                ],
                "scored_tokens": cell["metrics"][
                    "scored_tokens"
                ],
                "exact_records": cell["metrics"][
                    "exact_records"
                ],
            }
            reproduction_rows.append(
                {
                    "label": cell["label"],
                    "policy_id": cell["policy_id"],
                    "expected": expected,
                    "actual": actual,
                    "exact_match": actual == expected,
                }
            )
    reproduction_pass = (
        len(reproduction_rows) == 12
        and all(
            row["exact_match"] for row in reproduction_rows
        )
    )
    if not reproduction_pass:
        raise RuntimeError(
            "R4 failed to reproduce the original Finance target cell"
        )

    deltas: list[dict[str, Any]] = []
    for proposer_id in PROPOSER_IDS:
        entries = proposer_entries(plan, proposer_id)
        for entry in entries:
            policy_id = entry["policy_id"]
            finance = next(
                cell
                for cell in cells
                if cell["proposer_id"] == proposer_id
                and cell["condition_id"]
                == "finance_generation300_target_cut4"
                and cell["policy_id"] == policy_id
            )
            finance_records = per_record_lookup[
                (
                    proposer_id,
                    "finance_generation300_target_cut4",
                    policy_id,
                )
            ]
            for condition_id in (
                "public_base_target_cut4",
                "vikhr_heavy_target_cut4",
            ):
                other = next(
                    cell
                    for cell in cells
                    if cell["proposer_id"] == proposer_id
                    and cell["condition_id"] == condition_id
                    and cell["policy_id"] == policy_id
                )
                differences = paired_record_differences(
                    per_record_lookup[
                        (
                            proposer_id,
                            condition_id,
                            policy_id,
                        )
                    ],
                    finance_records,
                )
                deltas.append(
                    {
                        "proposer_id": proposer_id,
                        "policy_id": policy_id,
                        "label": entry["label"],
                        "condition_id": condition_id,
                        "reference_condition_id": (
                            "finance_generation300_target_cut4"
                        ),
                        "correct_token_delta": (
                            other["metrics"]["correct_tokens"]
                            - finance["metrics"][
                                "correct_tokens"
                            ]
                        ),
                        "token_accuracy_percentage_point_delta": (
                            100.0
                            * (
                                other["metrics"][
                                    "token_accuracy"
                                ]
                                - finance["metrics"][
                                    "token_accuracy"
                                ]
                            )
                        ),
                        "exact_record_delta": (
                            other["metrics"]["exact_records"]
                            - finance["metrics"]["exact_records"]
                        ),
                        "mean_record_accuracy_delta": (
                            statistics.mean(differences)
                        ),
                        "bootstrap_95": bootstrap_mean(
                            differences,
                            draws=10000,
                            seed=20260904,
                        ),
                        "better_records": sum(
                            value > 0 for value in differences
                        ),
                        "tied_records": sum(
                            value == 0 for value in differences
                        ),
                        "worse_records": sum(
                            value < 0 for value in differences
                        ),
                    }
                )

    leaders: list[dict[str, Any]] = []
    for proposer_id in PROPOSER_IDS:
        for condition_id in CONDITION_IDS:
            group = [
                cell
                for cell in cells
                if cell["proposer_id"] == proposer_id
                and cell["condition_id"] == condition_id
            ]
            ordered = sorted(
                group,
                key=lambda cell: (
                    cell["metrics"]["token_accuracy"],
                    cell["metrics"]["exact_records"],
                    -cell["cost"]["method_compute_seconds"],
                ),
                reverse=True,
            )
            baseline = next(
                (
                    cell
                    for cell in group
                    if cell["label"]
                    == "fixed_k256_direct"
                ),
                None,
            )
            for rank, cell in enumerate(ordered, start=1):
                cell["within_target_rank"] = rank
                if baseline is not None:
                    cell[
                        "runtime_relative_to_fixed_k256_direct"
                    ] = (
                        cell["cost"]["method_compute_seconds"]
                        / baseline["cost"][
                            "method_compute_seconds"
                        ]
                    )
                    cell[
                        "simulations_relative_to_fixed_k256_direct"
                    ] = (
                        cell["cost"]["candidate_simulations"]
                        / baseline["cost"][
                            "candidate_simulations"
                        ]
                    )
            winner = ordered[0]
            leaders.append(
                {
                    "proposer_id": proposer_id,
                    "condition_id": condition_id,
                    "label": winner["label"],
                    "policy_id": winner["policy_id"],
                    "correct_tokens": winner["metrics"][
                        "correct_tokens"
                    ],
                    "scored_tokens": winner["metrics"][
                        "scored_tokens"
                    ],
                    "token_accuracy": winner["metrics"][
                        "token_accuracy"
                    ],
                    "exact_records": winner["metrics"][
                        "exact_records"
                    ],
                    "records": winner["metrics"]["records"],
                    "method_compute_seconds": winner["cost"][
                        "method_compute_seconds"
                    ],
                    "candidate_simulations": winner["cost"][
                        "candidate_simulations"
                    ],
                }
            )

    result = {
        "schema": (
            "token-reconstruction.trr0002-owner-r4-"
            "historical-target-bridge-result.v1"
        ),
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "TARGET_ONLY_HISTORICAL_INPUT_BRIDGE_COMPLETE",
        "started_utc": started_utc,
        "truth_loaded_utc": started_utc,
        "ended_utc": utc_now(),
        "scoring_commit": git_head(root),
        "command": command_record(),
        "exit_status": 0,
        "one_variable_statement": (
            "All 128 exact historical token sequences, masks, "
            "positions, record order, cut depth, public A2 weights, "
            "A1 resource, policies, and scoring stay fixed within "
            "each paired comparison; only the unavailable target "
            "prefix weights change."
        ),
        "benchmark": {
            "records": EXPECTED_RECORDS,
            "positions": EXPECTED_POSITIONS,
            "valid_tokens_including_bos": (
                EXPECTED_VALID_WITH_BOS
            ),
            "scored_post_bos_tokens": (
                EXPECTED_SCORED_POST_BOS
            ),
            "truth_token_ids_sha256": tensor_sha256(truth),
            "attention_mask_sha256": tensor_sha256(mask),
            "position_ids_sha256": tensor_sha256(positions),
        },
        "reproduction_gate": {
            "status": "PASS",
            "all_12_exact": reproduction_pass,
            "rows": reproduction_rows,
        },
        "matrix": {
            "expected_cells": plan["matrix"]["expected_cells"],
            "completed_cells": len(cells),
            "cells": cells,
        },
        "leaders": leaders,
        "target_only_deltas_from_finance_generation300": (
            deltas
        ),
        "candidate_recall_curves": candidate_curves,
        "artifacts": {
            "plan": file_record(args.plan),
            "preparation_evidence": file_record(
                args.preparation_evidence
            ),
            "freeze_receipt": file_record(
                args.freeze_receipt
            ),
            "truth_sidecar": file_record(args.truth_sidecar),
            "prediction_outputs": {
                proposer_id: {
                    "predictions": file_record(
                        directory / "predictions.safetensors"
                    ),
                    "evidence": file_record(
                        directory / "evidence.json"
                    ),
                }
                for proposer_id, directory in directories.items()
            },
        },
        "claim_limits": plan["claim_limits"],
    }
    write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cells": len(cells),
                "reproduction": "PASS",
                "leaders": leaders,
            },
            sort_keys=True,
        )
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("R4 validation output is create-only")
    plan = load_json(args.plan)
    validate_plan(plan)
    preparation = load_json(args.preparation_evidence)
    receipt = load_json(args.freeze_receipt)
    result = load_json(args.result)
    checks = {
        "plan_frozen_before_target_generation": (
            plan["status"]
            == "FROZEN_BEFORE_TARGET_OBSERVATION_GENERATION"
        ),
        "exact_historical_records": (
            preparation["exact_input_identity"]["records"]
            == EXPECTED_RECORDS
        ),
        "exact_historical_positions": (
            preparation["exact_input_identity"]["positions"]
            == EXPECTED_POSITIONS
        ),
        "exact_post_bos_denominator": (
            preparation["exact_input_identity"][
                "scored_post_bos_tokens"
            ]
            == EXPECTED_SCORED_POST_BOS
        ),
        "three_target_conditions": (
            [
                item["condition_id"]
                for item in plan["targets"]
            ]
            == list(CONDITION_IDS)
        ),
        "only_target_weights_vary": all(
            plan["invariants"].values()
        ),
        "predictions_frozen_before_scoring": (
            receipt["status"]
            == "ALL_45_CELLS_FROZEN_BEFORE_SCORING"
        ),
        "truth_absent_from_prediction_processes": (
            receipt["truth_loaded_by_prediction_processes"]
            is False
        ),
        "target_calls_absent_from_prediction_processes": (
            receipt[
                "target_prefix_calls_by_prediction_processes"
            ]
            == 0
        ),
        "complete_matrix": (
            result["matrix"]["completed_cells"]
            == result["matrix"]["expected_cells"]
            == plan["matrix"]["expected_cells"]
        ),
        "original_finance_cell_reproduced": (
            result["reproduction_gate"]["all_12_exact"]
            is True
        ),
        "exact_input_hash_agreement": (
            result["benchmark"]["truth_token_ids_sha256"]
            == preparation["exact_input_identity"][
                "truth_token_ids_sha256"
            ]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"R4 validation failed: {checks}")
    output = {
        "schema": (
            "token-reconstruction.trr0002-owner-r4-validation.v1"
        ),
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "PASS",
        "created_utc": utc_now(),
        "command": command_record(),
        "exit_status": 0,
        "checks": checks,
        "artifacts": {
            "plan": file_record(args.plan),
            "preparation_evidence": file_record(
                args.preparation_evidence
            ),
            "freeze_receipt": file_record(
                args.freeze_receipt
            ),
            "result": file_record(args.result),
        },
    }
    write_json_exclusive(args.output, output)
    print(
        json.dumps(
            {"status": output["status"], "checks": len(checks)},
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    commands = {
        "preregister": command_preregister,
        "prepare": command_prepare,
        "predict": command_predict,
        "combine": command_combine,
        "freeze": command_freeze,
        "score": command_score,
        "validate": command_validate,
    }
    return commands[args.command_name](args)


if __name__ == "__main__":
    raise SystemExit(main())
