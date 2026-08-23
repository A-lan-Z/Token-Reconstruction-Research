#!/usr/bin/env python3
"""Owner-R3 strict-surrogate and heavy-fine-tune reconstruction study.

The workflow separates evaluator-only target preparation, isolated truthless
reconstruction, prediction freezing, and scoring.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import hmac
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import secrets
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
import transformers

from token_reconstruction.a1a2_configuration_search import (
    decode_policy,
    resolved_policy_from_dict,
)
from token_reconstruction.component_crossover import propose_public_a1
from token_reconstruction.dual_benchmark import (
    BOS_TOKEN_ID,
    scored_mask,
    validate_observations,
)
from token_reconstruction.experiment_runtime import (
    command_record,
    peak_memory,
    seed_everything,
)
from token_reconstruction.public_prefix import ContiguousPublicPrefix
from token_reconstruction.strict_base_surrogate import (
    canonical_mapping_bytes,
    exact_input_summary,
    isolated_record_batch_size,
    length_stratified_summary,
    propose_checkpoint_identity,
    right_padded_position_ids,
    sha256_text,
)


TASK_ID = "TRR-0002"
REVISION_ID = "TRR-0002-OWNER-REVISION-R3"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
BASE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
HEAVY_MODEL_ID = "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
HEAVY_REVISION = "7fa9d06a59246629244cdd3b6b92e4fc756baa0f"
HEAVY_DATASET_ID = "Vikhrmodels/GrandMaster-PRO-MAX"
HEAVY_DATASET_REVISION = "de9cc765d834f6d14f03155fe9c78b0b6c992b4c"
HEAVY_DATASET_FINGERPRINT = "fe2673b6a83b6691"
HISTORICAL_LENS_SHA256 = "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742"
FINANCE_CONFIG_SHA256 = "79ef9c9ca7400f003b94950dc299a8f3c314073258c76432ef273a920b3a460c"
FINANCE_TRACE_SHA256 = "0c880019a7b497fbf5d784b2cddb54f67b740c58c505fed8895697a1a1b4477a"
LENGTH_BINS = ((16, 32), (33, 64), (65, 96), (97, 128))
RECORDS_PER_BIN = 16
CONDITION_IDS = (
    "clean_pile_lora_cut4",
    "historical_finance_cut4",
    "grandmaster_public_base_cut4",
    "grandmaster_vikhr_heavy_cut4",
)
PROPOSER_IDS = ("checkpoint_identity", "alpaca_affine_control")


def fixed_policy(label: str, policy_id: str, k: int, score_rule: str) -> dict[str, Any]:
    return {
        "label": label,
        "policy_id": policy_id,
        "policy": {
            "numeric_thresholds": {},
            "policy_id": policy_id,
            "spec": {
                "fast_path_id": "off",
                "fast_path_threshold": None,
                "gate_comparator": "ge",
                "gate_mode": None,
                "kind": "fixed",
                "routing_signal": None,
                "schedule": [k],
                "score_rule": score_rule,
                "terminal_action": "commit_last_winner",
            },
        },
    }


POLICY_ENTRIES = (
    fixed_policy("fixed_k64_direct", "a1a2_589f6e179eb4626877c2", 64, "direct_cosine"),
    fixed_policy("fixed_k256_direct", "a1a2_43ea0bb737bc075531ca", 256, "direct_cosine"),
    fixed_policy(
        "fixed_k512_centered",
        "a1a2_13f73c306bf8946e9a28",
        512,
        "group_centered_cosine",
    ),
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


def import_path(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(
        name, path.resolve(strict=True)
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def require_file_record(expected: Mapping[str, Any], path: Path, label: str) -> None:
    actual = file_record(path)
    for field in ("bytes", "sha256"):
        if actual[field] != expected.get(field):
            raise RuntimeError(f"{label} {field} changed after freeze")


def validate_policies(entries: Sequence[Mapping[str, Any]]) -> None:
    if list(entries) != list(POLICY_ENTRIES):
        raise RuntimeError("R3 policy registry changed")
    for entry in entries:
        policy = resolved_policy_from_dict(entry["policy"])
        if policy.policy_id != entry["policy_id"]:
            raise RuntimeError("R3 policy identifier changed")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command_name", required=True)

    preregister = commands.add_parser("preregister")
    preregister.add_argument("--repository-root", type=Path, default=Path("."))
    preregister.add_argument("--request", type=Path, required=True)
    preregister.add_argument("--base-model-path", type=Path, required=True)
    preregister.add_argument("--heavy-model-path", type=Path, required=True)
    preregister.add_argument("--historical-lens", type=Path, required=True)
    preregister.add_argument("--selection-key", type=Path, required=True)
    preregister.add_argument("--output", type=Path, required=True)

    heavy = commands.add_parser("prepare-heavy")
    heavy.add_argument("--repository-root", type=Path, default=Path("."))
    heavy.add_argument("--plan", type=Path, required=True)
    heavy.add_argument("--selection-key", type=Path, required=True)
    heavy.add_argument("--base-model-path", type=Path, required=True)
    heavy.add_argument("--heavy-model-path", type=Path, required=True)
    heavy.add_argument("--input-root", type=Path, required=True)
    heavy.add_argument("--truth-sidecar", type=Path, required=True)
    heavy.add_argument("--selection-commitment", type=Path, required=True)
    heavy.add_argument("--preparation-evidence", type=Path, required=True)

    canonical = commands.add_parser("prepare-canonical")
    canonical.add_argument("--repository-root", type=Path, default=Path("."))
    canonical.add_argument("--plan", type=Path, required=True)
    canonical.add_argument("--historical-root", type=Path, required=True)
    canonical.add_argument("--clean-input-root", type=Path, required=True)
    canonical.add_argument("--input-root", type=Path, required=True)
    canonical.add_argument("--preparation-evidence", type=Path, required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--input-root", type=Path, required=True)
    predict.add_argument("--model-path", type=Path, required=True)
    predict.add_argument("--proposer", choices=PROPOSER_IDS, required=True)
    predict.add_argument("--condition-id", choices=CONDITION_IDS)
    predict.add_argument("--lens-path", type=Path)
    predict.add_argument("--output-directory", type=Path, required=True)
    predict.add_argument("--access-manifest", type=Path, required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--repository-root", type=Path, default=Path("."))
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--canonical-input-root", type=Path, required=True)
    freeze.add_argument("--heavy-input-root", type=Path, required=True)
    freeze.add_argument(
        "--prediction-directory", action="append", type=Path, required=True
    )
    freeze.add_argument("--output", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--repository-root", type=Path, default=Path("."))
    score.add_argument("--historical-root", type=Path, required=True)
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--freeze-receipt", type=Path, required=True)
    score.add_argument("--selection-key", type=Path, required=True)
    score.add_argument("--selection-commitment", type=Path, required=True)
    score.add_argument("--heavy-truth", type=Path, required=True)
    score.add_argument("--heavy-preparation", type=Path, required=True)
    score.add_argument("--clean-truth-jsonl", type=Path, required=True)
    score.add_argument("--selection-reveal", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--freeze-receipt", type=Path, required=True)
    validate.add_argument("--selection-commitment", type=Path, required=True)
    validate.add_argument("--selection-reveal", type=Path, required=True)
    validate.add_argument("--result", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    return root


def snapshot_artifacts(
    path: Path, *, include_readme: bool
) -> dict[str, Any]:
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    if include_readme:
        required.append("README.md")
    return {name: file_record(path / name) for name in required}


def command_preregister(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve(strict=True)
    request = args.request.resolve(strict=True)
    base_path = args.base_model_path.resolve(strict=True)
    heavy_path = args.heavy_model_path.resolve(strict=True)
    lens = args.historical_lens.resolve(strict=True)
    if args.output.exists() or args.selection_key.exists():
        raise RuntimeError("R3 preregistration outputs are create-only")
    if sha256_file(lens) != HISTORICAL_LENS_SHA256:
        raise RuntimeError("historical control lens identity changed")

    base_artifacts = snapshot_artifacts(base_path, include_readme=False)
    heavy_artifacts = snapshot_artifacts(heavy_path, include_readme=True)
    if (
        base_artifacts["tokenizer.json"]["sha256"]
        != heavy_artifacts["tokenizer.json"]["sha256"]
    ):
        raise RuntimeError("heavy target tokenizer differs from public base")
    base_config = load_json(base_path / "config.json")
    heavy_config = load_json(heavy_path / "config.json")
    architecture_fields = (
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "bos_token_id",
    )
    architecture_match = {
        field: base_config.get(field) == heavy_config.get(field)
        for field in architecture_fields
    }
    if not all(architecture_match.values()):
        raise RuntimeError("heavy target architecture differs from public base")
    if heavy_config.get("_name_or_path") != BASE_MODEL_ID:
        raise RuntimeError("heavy target config does not name the required base")
    card = (heavy_path / "README.md").read_text(encoding="utf-8")
    if BASE_MODEL_ID not in card or "150k instructions" not in card:
        raise RuntimeError("model card lacks lineage or training-scale evidence")

    from huggingface_hub import HfApi

    api = HfApi()
    base_info = api.model_info(BASE_MODEL_ID, revision=BASE_REVISION)
    model_info = api.model_info(HEAVY_MODEL_ID, revision=HEAVY_REVISION)
    dataset_info = api.dataset_info(
        HEAVY_DATASET_ID, revision=HEAVY_DATASET_REVISION
    )
    if (
        base_info.sha != BASE_REVISION
        or model_info.sha != HEAVY_REVISION
        or dataset_info.sha != HEAVY_DATASET_REVISION
    ):
        raise RuntimeError("live public metadata revision changed")
    card_data = model_info.card_data.to_dict() if model_info.card_data else {}
    declared_base = card_data.get("base_model")
    if declared_base not in (BASE_MODEL_ID, [BASE_MODEL_ID]):
        raise RuntimeError("base_model metadata does not match")

    key = secrets.token_bytes(32)
    args.selection_key.parent.mkdir(parents=True, exist_ok=True)
    with args.selection_key.open("xb") as handle:
        handle.write(key)
    os.chmod(args.selection_key, 0o600)
    matrix = [
        {
            "condition_id": condition,
            "proposer_id": proposer,
            "policy_id": entry["policy_id"],
            "method_id": f"{proposer}__{entry['label']}",
        }
        for condition in CONDITION_IDS
        for proposer in PROPOSER_IDS
        for entry in POLICY_ENTRIES
    ]
    payload = {
        "schema": "token-reconstruction.trr0002-owner-r3-preregistration.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "FROZEN_BEFORE_R3_TARGET_OBSERVATIONS",
        "created_utc": utc_now(),
        "created_from_commit": git_head(root),
        "owner_request": file_record(request),
        "scientific_question": (
            "How well does A1+A2 reconstruct when A1 has no fitted lens and uses "
            "only the untouched public checkpoint, and how robust is it to a much "
            "heavier same-base instruction fine-tune?"
        ),
        "terminology": {
            "official_checkpoint": BASE_MODEL_ID,
            "owner_shorthand": "Llama-3-1B-Instruct",
            "correction": (
                "Meta's official one-billion-parameter instruction checkpoint "
                "is Llama-3.2-1B-Instruct; no Llama-3.1 1B checkpoint is used."
            ),
        },
        "lineage": {
            "base": {
                "model_id": BASE_MODEL_ID,
                "revision": BASE_REVISION,
                "artifacts": base_artifacts,
                "model_card_url": f"https://huggingface.co/{BASE_MODEL_ID}",
                "api_revision_verified": base_info.sha,
            },
            "heavy_target": {
                "model_id": HEAVY_MODEL_ID,
                "revision": HEAVY_REVISION,
                "declared_base_model": declared_base,
                "config_name_or_path": heavy_config["_name_or_path"],
                "architecture_match": architecture_match,
                "artifacts": heavy_artifacts,
                "fine_tuning": {
                    "algorithm": "supervised fine-tuning",
                    "dataset": HEAVY_DATASET_ID,
                    "dataset_revision": HEAVY_DATASET_REVISION,
                    "model_card_instruction_count": 150000,
                    "model_card_url": f"https://huggingface.co/{HEAVY_MODEL_ID}",
                    "dataset_card_url": (
                        f"https://huggingface.co/datasets/{HEAVY_DATASET_ID}"
                    ),
                },
                "popularity_snapshot": {
                    "downloads_last_month": model_info.downloads,
                    "likes": model_info.likes,
                    "observed_utc": utc_now(),
                    "api_url": (
                        f"https://huggingface.co/api/models/{HEAVY_MODEL_ID}"
                    ),
                },
            },
            "dataset_snapshot": {
                "dataset_id": HEAVY_DATASET_ID,
                "revision": HEAVY_DATASET_REVISION,
                "downloads_last_month": dataset_info.downloads,
                "likes": dataset_info.likes,
                "split": "test",
                "expected_fingerprint": HEAVY_DATASET_FINGERPRINT,
            },
            "same_tokenizer_sha256": base_artifacts["tokenizer.json"]["sha256"],
        },
        "selection": {
            "language": "ru",
            "role": "first user message",
            "split": "test",
            "length_including_bos_bins": [list(value) for value in LENGTH_BINS],
            "records_per_bin": RECORDS_PER_BIN,
            "records": 64,
            "algorithm": (
                "within each bin sort eligible rows by HMAC-SHA256 of the secret "
                "key, dataset revision, row index, and source-text SHA-256; take "
                "the first 16; expose only opaque sequential IDs before freeze"
            ),
            "selection_key_sha256": hashlib.sha256(key).hexdigest(),
            "key_reveal": "only after all four prediction artifacts are frozen",
        },
        "surrogate": {
            "checkpoint": BASE_MODEL_ID,
            "revision": BASE_REVISION,
            "boundary_layer": 4,
            "prefix_layers": [0, 1, 2, 3],
            "dtype": "bfloat16",
            "identity_proposer": {
                "fitted_parameters": 0,
                "auxiliary_training_rows": 0,
                "rule": (
                    "cosine rank observed activation directly against the "
                    "public checkpoint input embeddings"
                ),
                "confidence_temperature_log": 3.0,
                "confidence_used_by_policies": False,
            },
            "alpaca_control": {
                "role": "historical fitted-lens control only",
                "lens": file_record(lens),
            },
            "target_weights_available_to_reconstruction": False,
            "target_prefix_calls_permitted": 0,
        },
        "policies": list(POLICY_ENTRIES),
        "conditions": [
            {"condition_id": CONDITION_IDS[0], "role": "canonical clean retrospective"},
            {"condition_id": CONDITION_IDS[1], "role": "historical Finance retrospective"},
            {"condition_id": CONDITION_IDS[2], "role": "fresh matched public-base control"},
            {"condition_id": CONDITION_IDS[3], "role": "fresh heavy-fine-tuned target"},
        ],
        "matrix": {
            "proposers": list(PROPOSER_IDS),
            "policies": 3,
            "conditions": 4,
            "required_cells": 24,
            "cells": matrix,
        },
        "primary_comparisons": [
            "checkpoint identity fixed K256 direct: heavy target versus matched base",
            "checkpoint identity versus Alpaca control at K256 within every condition",
        ],
        "secondary_comparisons": [
            "K64 direct cost point",
            "K512 group-centered target-shift point",
        ],
        "metrics": [
            "post-BOS token accuracy",
            "exact token-complete inputs",
            "exact decoded-text inputs",
            "exact original-source strings for the heavy panel",
            "errors among failed inputs and first-error position",
            "length-stratified exact and token accuracy",
            "proposal recall and conditional selector accuracy",
            "runtime, simulations, GPU memory, and process RSS",
        ],
        "execution": {
            "heavy_target_and_truth_prepared_evaluator_side": True,
            "prediction_processes": 4,
            "isolation": "user, mount, network, and PID namespaces with pivot_root",
            "strict_process_lens_mounted": False,
            "control_process_only_lens_mounted": True,
            "freeze_before_truth": True,
            "canonical_truth_status": "already open retrospective",
            "heavy_truth_status": "hidden until prediction freeze",
        },
        "code": {
            "runner": file_record(Path(__file__)),
            "strict_module": file_record(
                root / "src/token_reconstruction/strict_base_surrogate.py"
            ),
            "configuration_engine": file_record(
                root / "src/token_reconstruction/a1a2_configuration_search.py"
            ),
        },
        "claim_scope": (
            "The heavy panel supports only a same-model, same-prompts target-shift "
            "claim; replacement language additionally requires both canonical cells."
        ),
    }
    validate_policies(payload["policies"])
    write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "cells": len(matrix), "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


def validate_plan(plan: Mapping[str, Any], key_path: Path | None = None) -> None:
    if (
        plan.get("schema")
        != "token-reconstruction.trr0002-owner-r3-preregistration.v1"
        or plan.get("status") != "FROZEN_BEFORE_R3_TARGET_OBSERVATIONS"
        or plan.get("task_id") != TASK_ID
        or plan.get("revision_id") != REVISION_ID
        or plan["matrix"]["required_cells"] != 24
    ):
        raise RuntimeError("R3 preregistration is invalid")
    validate_policies(plan["policies"])
    if key_path is not None:
        key = key_path.read_bytes()
        if (
            len(key) != 32
            or hashlib.sha256(key).hexdigest()
            != plan["selection"]["selection_key_sha256"]
        ):
            raise RuntimeError("R3 selection key differs from preregistration")


def first_user_text(row: Mapping[str, Any]) -> str | None:
    conversation = row.get("conversation")
    if not isinstance(conversation, list):
        return None
    for message in conversation:
        if isinstance(message, Mapping) and message.get("role") == "user":
            value = message.get("content")
            return value if isinstance(value, str) and value else None
    return None


def select_heavy_records(
    plan: Mapping[str, Any], key: bytes, tokenizer: Any
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        HEAVY_DATASET_ID,
        revision=HEAVY_DATASET_REVISION,
        split="test",
    )
    if dataset._fingerprint != HEAVY_DATASET_FINGERPRINT:
        raise RuntimeError(f"heavy dataset fingerprint changed: {dataset._fingerprint}")
    by_bin: dict[str, list[dict[str, Any]]] = {
        f"{lower}-{upper}": [] for lower, upper in LENGTH_BINS
    }
    for index, row in enumerate(dataset):
        if row.get("prompt_lang") != "ru":
            continue
        text_value = first_user_text(row)
        if text_value is None:
            continue
        ids = tokenizer.encode(text_value, add_special_tokens=True)
        if not ids or ids[0] != BOS_TOKEN_ID or len(ids) > 128:
            continue
        for lower, upper in LENGTH_BINS:
            if lower <= len(ids) <= upper:
                label = f"{lower}-{upper}"
                source_hash = sha256_text(text_value)
                message = (
                    HEAVY_DATASET_REVISION.encode("ascii")
                    + b"\0"
                    + str(index).encode("ascii")
                    + b"\0"
                    + source_hash.encode("ascii")
                )
                by_bin[label].append(
                    {
                        "dataset_index": index,
                        "source_text": text_value,
                        "source_sha256": source_hash,
                        "token_ids": ids,
                        "token_length": len(ids),
                        "length_bin": label,
                        "selection_digest": hmac.new(
                            key, message, hashlib.sha256
                        ).hexdigest(),
                    }
                )
                break
    selected: list[dict[str, Any]] = []
    for lower, upper in LENGTH_BINS:
        label = f"{lower}-{upper}"
        candidates = sorted(
            by_bin[label],
            key=lambda row: (row["selection_digest"], row["dataset_index"]),
        )
        if len(candidates) < RECORDS_PER_BIN:
            raise RuntimeError(f"not enough eligible rows in length bin {label}")
        selected.extend(candidates[:RECORDS_PER_BIN])
    for index, row in enumerate(selected, start=1):
        row["opaque_id"] = f"heavy-r3-{index:06d}"
    return selected


@torch.inference_mode()
def generate_prefix_observations(
    model_path: Path,
    token_rows: Sequence[Sequence[int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
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
    output = torch.zeros((len(token_rows), 128, 2048), dtype=torch.bfloat16)
    for index, ids in enumerate(token_rows):
        inputs = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
        hidden = prefix.forward_full(inputs)
        output[index, : len(ids)] = hidden[0].detach().cpu()
        del inputs, hidden
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    del prefix, model
    gc.collect()
    torch.cuda.empty_cache()
    return output, elapsed


def model_weight_drift(base_file: Path, target_file: Path) -> dict[str, Any]:
    with safe_open(base_file, framework="pt", device="cpu") as base, safe_open(
        target_file, framework="pt", device="cpu"
    ) as target:
        base_keys = list(base.keys())
        if base_keys != list(target.keys()):
            raise RuntimeError("base and heavy target tensor registries differ")
        total_values = changed_values = changed_tensors = 0
        prefix_values = 0
        delta_sq = base_sq = prefix_delta_sq = prefix_base_sq = 0.0
        maximum_absolute_delta = 0.0
        for name in base_keys:
            left = base.get_tensor(name).reshape(-1)
            right = target.get_tensor(name).reshape(-1)
            if left.shape != right.shape:
                raise RuntimeError(f"weight shape changed: {name}")
            tensor_changed = False
            is_prefix = name == "model.embed_tokens.weight" or any(
                name.startswith(f"model.layers.{index}.") for index in range(4)
            )
            for start in range(0, left.numel(), 4_000_000):
                stop = min(start + 4_000_000, left.numel())
                x = left[start:stop].float()
                y = right[start:stop].float()
                delta = y - x
                count = int(delta.ne(0).sum().item())
                changed_values += count
                tensor_changed |= count > 0
                total_values += int(delta.numel())
                block_delta = float(torch.sum(delta.double().square()).item())
                block_base = float(torch.sum(x.double().square()).item())
                delta_sq += block_delta
                base_sq += block_base
                maximum_absolute_delta = max(
                    maximum_absolute_delta, float(delta.abs().max().item())
                )
                if is_prefix:
                    prefix_values += int(delta.numel())
                    prefix_delta_sq += block_delta
                    prefix_base_sq += block_base
                del x, y, delta
            changed_tensors += int(tensor_changed)
    return {
        "tensor_count": len(base_keys),
        "changed_tensors": changed_tensors,
        "parameter_values": total_values,
        "changed_parameter_values": changed_values,
        "changed_parameter_fraction": changed_values / total_values,
        "relative_l2_all_weights": math.sqrt(delta_sq / base_sq),
        "prefix_parameter_values": prefix_values,
        "relative_l2_embedding_and_layers_0_to_3": math.sqrt(
            prefix_delta_sq / prefix_base_sq
        ),
        "maximum_absolute_parameter_delta": maximum_absolute_delta,
    }


def command_prepare_heavy(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve(strict=True)
    plan = load_json(args.plan)
    validate_plan(plan, args.selection_key)
    output_paths = (
        args.truth_sidecar,
        args.selection_commitment,
        args.preparation_evidence,
        args.input_root / "config.json",
        args.input_root / "observations.safetensors",
    )
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise RuntimeError("heavy preparation outputs are create-only")
    base_path = args.base_model_path.resolve(strict=True)
    heavy_path = args.heavy_model_path.resolve(strict=True)
    for name, record in plan["lineage"]["base"]["artifacts"].items():
        require_file_record(record, base_path / name, f"base {name}")
    for name, record in plan["lineage"]["heavy_target"]["artifacts"].items():
        require_file_record(record, heavy_path / name, f"heavy {name}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    key = args.selection_key.read_bytes()
    selected = select_heavy_records(plan, key, tokenizer)
    mapping = [
        {
            "opaque_id": row["opaque_id"],
            "dataset_index": row["dataset_index"],
            "source_sha256": row["source_sha256"],
            "token_length": row["token_length"],
            "length_bin": row["length_bin"],
        }
        for row in selected
    ]
    selection_hmac = hmac.new(
        key, canonical_mapping_bytes(mapping), hashlib.sha256
    ).hexdigest()
    token_rows = [row["token_ids"] for row in selected]
    truth = {
        "schema": "token-reconstruction.trr0002-owner-r3-heavy-truth.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "plan_sha256": sha256_file(args.plan),
        "selection_hmac_sha256": selection_hmac,
        "records": [
            {
                **mapping[index],
                "source_text": row["source_text"],
                "token_ids": row["token_ids"],
            }
            for index, row in enumerate(selected)
        ],
    }
    write_json_exclusive(args.truth_sidecar, truth)
    os.chmod(args.truth_sidecar, 0o600)
    attention_mask = torch.zeros((64, 128), dtype=torch.long)
    for index, ids in enumerate(token_rows):
        attention_mask[index, : len(ids)] = 1
    position_ids = right_padded_position_ids(attention_mask)

    if not torch.cuda.is_available():
        raise RuntimeError("heavy target preparation requires CUDA")
    device = torch.device("cuda")
    seed_everything(20260901)
    started_utc = utc_now()
    torch.cuda.reset_peak_memory_stats(device)
    base_observations, base_seconds = generate_prefix_observations(
        base_path, token_rows, device=device
    )
    heavy_observations, heavy_seconds = generate_prefix_observations(
        heavy_path, token_rows, device=device
    )
    validate_observations(base_observations, attention_mask, position_ids)
    validate_observations(
        heavy_observations, attention_mask, position_ids
    )
    drift = model_weight_drift(
        base_path / "model.safetensors", heavy_path / "model.safetensors"
    )

    args.input_root.mkdir(parents=True, exist_ok=False)
    observations_path = args.input_root / "observations.safetensors"
    save_file(
        {
            "grandmaster_public_base_cut4.activations": base_observations.contiguous(),
            "grandmaster_public_base_cut4.attention_mask": (
                attention_mask.to(torch.uint8).contiguous()
            ),
            "grandmaster_public_base_cut4.position_ids": (
                position_ids.to(torch.int32).contiguous()
            ),
            "grandmaster_vikhr_heavy_cut4.activations": heavy_observations.contiguous(),
            "grandmaster_vikhr_heavy_cut4.attention_mask": (
                attention_mask.to(torch.uint8).contiguous()
            ),
            "grandmaster_vikhr_heavy_cut4.position_ids": (
                position_ids.to(torch.int32).contiguous()
            ),
        },
        observations_path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r3-heavy-observations.v1",
            "truth_included": "false",
            "selection_hmac_sha256": selection_hmac,
            "target_prefix_calls": "128",
        },
    )
    config = {
        "schema": "token-reconstruction.trr0002-owner-r3-sanitized-input.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "plan_sha256": sha256_file(args.plan),
        "selection_hmac_sha256": selection_hmac,
        "truth_included": False,
        "dataset_indices_included": False,
        "source_text_or_hash_included": False,
        "conditions": [
            {
                "condition_id": "grandmaster_public_base_cut4",
                "records": 64,
                "positions": 128,
                "hidden_size": 2048,
                "target": "matched public checkpoint",
            },
            {
                "condition_id": "grandmaster_vikhr_heavy_cut4",
                "records": 64,
                "positions": 128,
                "hidden_size": 2048,
                "target": f"{HEAVY_MODEL_ID}@{HEAVY_REVISION}",
            },
        ],
        "opaque_record_ids": [row["opaque_id"] for row in selected],
        "policies": list(POLICY_ENTRIES),
        "base_model": {
            "id": BASE_MODEL_ID,
            "revision": BASE_REVISION,
            "model_safetensors_sha256": (
                plan["lineage"]["base"]["artifacts"]["model.safetensors"]["sha256"]
            ),
        },
        "permitted_files": ["config.json", "observations.safetensors"],
    }
    write_json_exclusive(args.input_root / "config.json", config)
    commitment = {
        "schema": "token-reconstruction.trr0002-owner-r3-heavy-selection-commitment.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "HIDDEN_MAPPING_COMMITTED_BEFORE_RECONSTRUCTION",
        "created_utc": utc_now(),
        "plan": file_record(args.plan),
        "selection_key_sha256": plan["selection"]["selection_key_sha256"],
        "selection_hmac_sha256": selection_hmac,
        "records": 64,
        "length_bins": {
            f"{lower}-{upper}": RECORDS_PER_BIN for lower, upper in LENGTH_BINS
        },
        "sanitized_config": file_record(args.input_root / "config.json"),
        "sanitized_observations": file_record(observations_path),
        "mapping_or_source_disclosed": False,
    }
    write_json_exclusive(args.selection_commitment, commitment)
    evidence = {
        "schema": "token-reconstruction.trr0002-owner-r3-heavy-preparation.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "EVALUATOR_TARGET_PREPARATION_COMPLETE",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "execution_commit": git_head(root),
        "command": command_record(),
        "plan": file_record(args.plan),
        "selection_commitment": file_record(args.selection_commitment),
        "truth_sidecar": {
            **file_record(args.truth_sidecar),
            "retention": "evaluator-private until prediction freeze",
        },
        "target_prefix_execution": {
            "public_base_calls": 64,
            "heavy_target_calls": 64,
            "public_base_seconds": base_seconds,
            "heavy_target_seconds": heavy_seconds,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "boundary_layer": 4,
        },
        "model_weight_drift": drift,
        "tokenizer_byte_identical": True,
        "lengths": {
            "minimum": min(len(ids) for ids in token_rows),
            "median": statistics.median(len(ids) for ids in token_rows),
            "maximum": max(len(ids) for ids in token_rows),
            "mean": statistics.mean(len(ids) for ids in token_rows),
            "scored_tokens": int(attention_mask[:, 1:].sum().item()),
            "bins": {
                f"{lower}-{upper}": sum(
                    lower <= len(ids) <= upper for ids in token_rows
                )
                for lower, upper in LENGTH_BINS
            },
        },
        "observations": file_record(observations_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "peak_memory": peak_memory(),
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    write_json_exclusive(args.preparation_evidence, evidence)
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "records": 64,
                "scored_tokens": evidence["lengths"]["scored_tokens"],
                "selection_hmac": selection_hmac,
            },
            sort_keys=True,
        )
    )
    return 0


def historical_observations(
    historical_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], Any, list[Any]]:
    source_path = historical_root / "scripts/score_a1_a2_source300_20260809.py"
    source300 = import_path("trr0002_r3_source300", source_path)
    config_path = historical_root / "config/a1_a2_source300_static_20260809.json"
    if sha256_file(config_path) != FINANCE_CONFIG_SHA256:
        raise RuntimeError("Finance source config changed")
    config = source300.load_config(config_path)
    trace_path = source300.resolve_inside_ersoy(config["source"]["path"])
    if sha256_file(trace_path) != FINANCE_TRACE_SHA256:
        raise RuntimeError("Finance activation trace changed")
    captures = source300.load_source_payload(trace_path, config)
    observations = torch.cat(
        [value.activation.detach().cpu() for value in captures], dim=0
    ).contiguous()
    mask = torch.cat(
        [value.attention_mask.detach().cpu().long() for value in captures], dim=0
    ).contiguous()
    positions = torch.cat(
        [value.position_ids.detach().cpu().long() for value in captures], dim=0
    ).contiguous()
    validate_observations(observations, mask, positions)
    return observations, mask, positions, config, source300, captures


def command_prepare_canonical(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve(strict=True)
    plan = load_json(args.plan)
    validate_plan(plan)
    output_paths = (
        args.input_root / "config.json",
        args.input_root / "observations.safetensors",
        args.preparation_evidence,
    )
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise RuntimeError("canonical preparation outputs are create-only")
    clean_path = (
        args.clean_input_root.resolve(strict=True)
        / "observations/unavailable_target_lora_cut4.safetensors"
    )
    clean = load_file(clean_path, device="cpu")["activations"].contiguous()
    clean_mask = torch.ones((64, 40), dtype=torch.long)
    clean_positions = torch.arange(40).view(1, -1).expand(64, -1).contiguous()
    validate_observations(clean, clean_mask, clean_positions)
    finance, finance_mask, finance_positions, config, _, _ = historical_observations(
        args.historical_root.resolve(strict=True)
    )
    args.input_root.mkdir(parents=True, exist_ok=False)
    observations_path = args.input_root / "observations.safetensors"
    save_file(
        {
            "clean_pile_lora_cut4.activations": clean,
            "clean_pile_lora_cut4.attention_mask": clean_mask.to(torch.uint8),
            "clean_pile_lora_cut4.position_ids": clean_positions.to(torch.int32),
            "historical_finance_cut4.activations": finance,
            "historical_finance_cut4.attention_mask": finance_mask.to(torch.uint8),
            "historical_finance_cut4.position_ids": finance_positions.to(torch.int32),
        },
        observations_path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r3-canonical-observations.v1",
            "truth_included": "false",
            "truth_status": "already-open-retrospective",
        },
    )
    config_payload = {
        "schema": "token-reconstruction.trr0002-owner-r3-sanitized-input.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "plan_sha256": sha256_file(args.plan),
        "truth_included": False,
        "dataset_indices_included": False,
        "source_text_or_hash_included": False,
        "conditions": [
            {
                "condition_id": "clean_pile_lora_cut4",
                "records": 64,
                "positions": 40,
                "hidden_size": 2048,
                "target": "canonical unavailable rank-4 Pile LoRA",
            },
            {
                "condition_id": "historical_finance_cut4",
                "records": 128,
                "positions": 128,
                "hidden_size": 2048,
                "target": "generation-300 Finance-Instruct target",
            },
        ],
        "opaque_record_ids": {
            "clean_pile_lora_cut4": [
                f"canonical-clean-{index:06d}" for index in range(1, 65)
            ],
            "historical_finance_cut4": [
                f"historical-finance-{index:06d}" for index in range(1, 129)
            ],
        },
        "policies": list(POLICY_ENTRIES),
        "base_model": {
            "id": BASE_MODEL_ID,
            "revision": BASE_REVISION,
            "model_safetensors_sha256": (
                plan["lineage"]["base"]["artifacts"]["model.safetensors"]["sha256"]
            ),
        },
        "permitted_files": ["config.json", "observations.safetensors"],
    }
    write_json_exclusive(args.input_root / "config.json", config_payload)
    evidence = {
        "schema": "token-reconstruction.trr0002-owner-r3-canonical-preparation.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "SANITIZED_CANONICAL_OBSERVATIONS_PREPARED_WITHOUT_TRUTH",
        "created_utc": utc_now(),
        "execution_commit": git_head(root),
        "command": command_record(),
        "plan": file_record(args.plan),
        "config": file_record(args.input_root / "config.json"),
        "observations": file_record(observations_path),
        "sources": {
            "clean_observation": file_record(clean_path),
            "finance_source_config": file_record(
                args.historical_root
                / "config/a1_a2_source300_static_20260809.json"
            ),
            "finance_target": {
                "checkpoint_generation": config["source"]["checkpoint_generation"],
                "weight_version": config["source"]["weight_version"],
            },
        },
        "truth_loaded": False,
    }
    write_json_exclusive(args.preparation_evidence, evidence)
    print(json.dumps({"status": evidence["status"], "conditions": 2}, sort_keys=True))
    return 0


def load_public_surrogate(
    model_path: Path,
) -> tuple[ContiguousPublicPrefix, torch.Tensor, torch.device]:
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("R3 reconstruction requires CUDA")
    device = torch.device("cuda")
    full = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    full.requires_grad_(False)
    if (
        int(full.config.hidden_size) != 2048
        or int(full.config.vocab_size) != 128256
        or len(full.model.layers) != 16
    ):
        raise RuntimeError("public surrogate architecture changed")
    prefix = ContiguousPublicPrefix(full, cut_depth=4).to(device).eval()
    embeddings = F.normalize(prefix.embed_tokens.weight.detach().float(), dim=-1)
    embeddings = torch.nan_to_num(embeddings).to(device)
    del full
    gc.collect()
    torch.cuda.empty_cache()
    return prefix, embeddings, device


def load_conditions(
    input_root: Path,
    config: Mapping[str, Any],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    state = load_file(input_root / "observations.safetensors", device="cpu")
    expected: set[str] = set()
    output: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for condition in config["conditions"]:
        condition_id = condition["condition_id"]
        keys = {
            f"{condition_id}.activations",
            f"{condition_id}.attention_mask",
            f"{condition_id}.position_ids",
        }
        expected.update(keys)
        observations = state[f"{condition_id}.activations"].contiguous()
        mask = state[f"{condition_id}.attention_mask"].long().contiguous()
        positions = state[f"{condition_id}.position_ids"].long().contiguous()
        if observations.shape != (
            condition["records"],
            condition["positions"],
            condition["hidden_size"],
        ):
            raise RuntimeError(f"condition geometry changed: {condition_id}")
        validate_observations(observations, mask, positions)
        output[condition_id] = (observations, mask, positions)
    if set(state) != expected:
        raise RuntimeError("sanitized observation tensor registry changed")
    return output


def command_predict(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    if (
        config.get("schema")
        != "token-reconstruction.trr0002-owner-r3-sanitized-input.v1"
        or config.get("truth_included") is not False
        or config.get("dataset_indices_included") is not False
        or config.get("source_text_or_hash_included") is not False
    ):
        raise RuntimeError("sanitized prediction config is invalid")
    validate_policies(config["policies"])
    if args.config.resolve() != (args.input_root / "config.json").resolve():
        raise RuntimeError("prediction config must be inside sanitized input root")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if args.access_manifest.parent.resolve() != args.output_directory.resolve():
        raise RuntimeError("access manifest must be inside prediction output")
    visible_outputs = {
        path.resolve() for path in args.output_directory.iterdir()
    }
    if visible_outputs != {args.access_manifest.resolve()}:
        raise RuntimeError(
            "prediction output must contain only the fresh access manifest"
        )
    access = load_json(args.access_manifest)
    if (
        access.get("status") != "PASS"
        or access.get("proposer_id") != args.proposer
        or access.get("truth_paths_visible") != 0
        or access.get("dataset_content_visible") is not False
        or access.get("target_model_visible") is not False
        or access.get("network_default_route") is not False
    ):
        raise RuntimeError("R3 access manifest failed")
    if args.proposer == "checkpoint_identity" and args.lens_path is not None:
        raise RuntimeError("strict checkpoint-only proposer may not receive a lens")
    if args.proposer == "alpaca_affine_control" and args.lens_path is None:
        raise RuntimeError("Alpaca control requires the frozen lens")

    observations_by_condition = load_conditions(args.input_root, config)
    if args.condition_id is not None:
        if args.condition_id not in observations_by_condition:
            raise RuntimeError("condition filter is absent from sanitized input")
        observations_by_condition = {
            args.condition_id: observations_by_condition[args.condition_id]
        }
    prefix, embeddings, device = load_public_surrogate(args.model_path)
    lens = None
    if args.proposer == "alpaca_affine_control":
        reference_path = (
            Path("/code/reference/strict_bos/round001_teacher.py")
            if Path("/code").exists()
            else Path("reference/strict_bos/round001_teacher.py")
        )
        reference = import_path("trr0002_r3_reference", reference_path)
        lens = reference.load_frozen_lens(args.lens_path, device=device)

    started_utc = utc_now()
    seed_everything(20260902)
    record_batch_size = isolated_record_batch_size(args.condition_id)
    tensors: dict[str, torch.Tensor] = {}
    costs: dict[str, Any] = {}
    for condition_id, (observations, mask, positions) in observations_by_condition.items():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if args.proposer == "checkpoint_identity":
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
        tensors[f"{condition_id}.candidates"] = proposal.candidates.to(torch.int32)
        tensors[f"{condition_id}.confidence"] = proposal.top1_confidence.float()
        condition_costs: dict[str, Any] = {}
        for entry in POLICY_ENTRIES:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            policy = resolved_policy_from_dict(entry["policy"])
            result = decode_policy(
                observations=observations,
                attention_mask=mask,
                position_ids=positions,
                candidates=proposal.candidates,
                a1_confidence=proposal.top1_confidence,
                precut=prefix,
                device=device,
                policy=policy,
                record_batch_size=record_batch_size,
            )
            key = f"{condition_id}.{entry['policy_id']}"
            tensors[f"{key}.predictions"] = result.predictions.to(torch.int32)
            tensors[f"{key}.routes"] = result.routes.to(torch.int8)
            tensors[f"{key}.selected_k"] = result.selected_k.to(torch.int16)
            condition_costs[entry["policy_id"]] = {
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
                        "status": "R3_PREDICTION_PROGRESS",
                        "proposer": args.proposer,
                        "condition": condition_id,
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
                "valid_positions": int(scored_mask(mask).sum().item()),
            },
            "policies": condition_costs,
        }

    prediction_path = args.output_directory / "predictions.safetensors"
    evidence_path = args.output_directory / "evidence.json"
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        prediction_path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r3-predictions.v1",
            "task_id": TASK_ID,
            "revision_id": REVISION_ID,
            "proposer_id": args.proposer,
            "truth_loaded": "false",
            "target_prefix_calls": "0",
        },
    )
    evidence = {
        "schema": "token-reconstruction.trr0002-owner-r3-reconstruction-evidence.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "PREDICTIONS_CREATED_WITHOUT_TRUTH",
        "proposer_id": args.proposer,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "command": command_record(),
        "exit_status": 0,
        "config": file_record(args.config),
        "condition_ids": list(costs),
        "record_batch_size": record_batch_size,
        "observations": file_record(args.input_root / "observations.safetensors"),
        "access_manifest": file_record(args.access_manifest),
        "predictions": file_record(prediction_path),
        "access": {
            "truth_inputs": 0,
            "dataset_inputs": 0,
            "target_weights_available": False,
            "target_prefix_calls": 0,
            "public_surrogate_only": True,
            "fitted_lens_available": (
                args.proposer == "alpaca_affine_control"
            ),
        },
        "proposer": {
            "id": args.proposer,
            "fitted_parameters": (
                0 if args.proposer == "checkpoint_identity" else 4_196_353
            ),
            "auxiliary_training_rows": (
                0 if args.proposer == "checkpoint_identity" else 52_002
            ),
            "lens": (
                None if args.lens_path is None else file_record(args.lens_path)
            ),
        },
        "costs": costs,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "process_max_rss_kib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "peak_memory": peak_memory(),
        },
    }
    write_json_exclusive(evidence_path, evidence)
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "proposer": args.proposer,
                "conditions": len(costs),
                "prediction_sha256": evidence["predictions"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_freeze(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve(strict=True)
    plan = load_json(args.plan)
    validate_plan(plan)
    if len(args.prediction_directory) != 6:
        raise RuntimeError(
            "R3 freeze requires exactly six isolated prediction directories"
        )
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for directory in args.prediction_directory:
        evidence_path = directory / "evidence.json"
        prediction_path = directory / "predictions.safetensors"
        access_path = directory / "access_manifest.json"
        evidence = load_json(evidence_path)
        access = load_json(access_path)
        if evidence.get("record_batch_size") != 4:
            raise RuntimeError(
                "every final prediction entry must use record batch size four"
            )
        require_file_record(evidence["predictions"], prediction_path, "prediction")
        require_file_record(
            evidence["access_manifest"], access_path, "access manifest"
        )
        if "condition_ids" in evidence:
            condition_ids = tuple(evidence["condition_ids"])
        else:
            config = load_json(Path(evidence["config"]["resolved_path"]))
            condition_ids = tuple(
                row["condition_id"] for row in config["conditions"]
            )
        identity = (evidence["proposer_id"], condition_ids)
        if identity in seen:
            raise RuntimeError(f"duplicate prediction identity: {identity}")
        seen.add(identity)
        if access["status"] != "PASS" or evidence["access"]["truth_inputs"] != 0:
            raise RuntimeError("prediction access evidence failed")
        os.chmod(prediction_path, 0o444)
        os.chmod(evidence_path, 0o444)
        os.chmod(access_path, 0o444)
        entries.append(
            {
                "proposer_id": evidence["proposer_id"],
                "condition_ids": list(condition_ids),
                "directory": str(directory),
                "predictions": file_record(prediction_path),
                "evidence": file_record(evidence_path),
                "access_manifest": file_record(access_path),
            }
        )
    expected = {
        (proposer, ("clean_pile_lora_cut4", "historical_finance_cut4"))
        for proposer in PROPOSER_IDS
    } | {
        (proposer, (condition,))
        for proposer in PROPOSER_IDS
        for condition in (
            "grandmaster_public_base_cut4",
            "grandmaster_vikhr_heavy_cut4",
        )
    }
    if seen != expected:
        raise RuntimeError(f"prediction matrix incomplete before freeze: {seen}")
    receipt = {
        "schema": "token-reconstruction.trr0002-owner-r3-freeze-receipt.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "ALL_24_CELLS_FROZEN_BEFORE_HEAVY_TRUTH",
        "created_utc": utc_now(),
        "execution_commit": git_head(root),
        "plan": file_record(args.plan),
        "canonical_input": {
            "config": file_record(args.canonical_input_root / "config.json"),
            "observations": file_record(
                args.canonical_input_root / "observations.safetensors"
            ),
        },
        "heavy_input": {
            "config": file_record(args.heavy_input_root / "config.json"),
            "observations": file_record(
                args.heavy_input_root / "observations.safetensors"
            ),
        },
        "prediction_entries": sorted(
            entries, key=lambda row: (row["condition_ids"][0], row["proposer_id"])
        ),
        "required_cells": 24,
        "heavy_truth_opened": False,
        "selection_key_opened": False,
        "prediction_processes": 6,
        "operational_deviation": {
            "preregistered_processes": 4,
            "actual_processes": 6,
            "initial_record_batch_size": 8,
            "final_common_record_batch_size": 4,
            "reason": (
                "the 16 GB device could not complete heavy-target K512 at "
                "batch eight, and an exact diagnostic found K64 prediction "
                "differences between batches eight and four"
            ),
            "entire_matrix_rerun_consistently": True,
            "execution_setting_changed": True,
            "scientific_settings_changed": False,
        },
        "target_prefix_calls_by_reconstructors": 0,
    }
    write_json_exclusive(args.output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "entries": len(entries),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def load_clean_truth(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 64:
        raise RuntimeError("clean truth record count changed")
    truth = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long)
    if truth.shape != (64, 40):
        raise RuntimeError("clean truth geometry changed")
    return truth, torch.ones_like(truth), [str(row["record_id"]) for row in rows]


def load_heavy_truth(
    path: Path,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
    list[dict[str, Any]],
]:
    payload = load_json(path)
    if (
        payload.get("schema")
        != "token-reconstruction.trr0002-owner-r3-heavy-truth.v1"
    ):
        raise RuntimeError("heavy truth schema changed")
    rows = payload["records"]
    if len(rows) != 64:
        raise RuntimeError("heavy truth record count changed")
    truth = torch.zeros((64, 128), dtype=torch.long)
    mask = torch.zeros((64, 128), dtype=torch.long)
    for index, row in enumerate(rows):
        ids = torch.tensor(row["token_ids"], dtype=torch.long)
        truth[index, : ids.numel()] = ids
        mask[index, : ids.numel()] = 1
    return (
        truth,
        mask,
        [str(row["opaque_id"]) for row in rows],
        [str(row["source_text"]) for row in rows],
        rows,
    )


def prediction_entries(
    receipt: Mapping[str, Any],
) -> dict[
    tuple[str, str],
    tuple[dict[str, torch.Tensor], Mapping[str, Any]],
]:
    output: dict[
        tuple[str, str],
        tuple[dict[str, torch.Tensor], Mapping[str, Any]],
    ] = {}
    for entry in receipt["prediction_entries"]:
        prediction_path = Path(entry["predictions"]["resolved_path"])
        evidence_path = Path(entry["evidence"]["resolved_path"])
        access_path = Path(entry["access_manifest"]["resolved_path"])
        require_file_record(entry["predictions"], prediction_path, "frozen prediction")
        require_file_record(entry["evidence"], evidence_path, "frozen evidence")
        require_file_record(
            entry["access_manifest"], access_path, "frozen access"
        )
        evidence = load_json(evidence_path)
        state = load_file(prediction_path, device="cpu")
        for condition_id in entry["condition_ids"]:
            key = (condition_id, entry["proposer_id"])
            if key in output:
                raise RuntimeError("duplicate frozen prediction condition")
            output[key] = (state, evidence)
    return output


def score_cell(
    *,
    condition_id: str,
    proposer_id: str,
    entry: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    evidence: Mapping[str, Any],
    truth: torch.Tensor,
    mask: torch.Tensor,
    record_ids: Sequence[str],
    tokenizer: Any,
    source_texts: Sequence[str] | None,
) -> dict[str, Any]:
    policy_id = entry["policy_id"]
    predictions = state[f"{condition_id}.{policy_id}.predictions"].long()
    candidates = state[f"{condition_id}.candidates"].long()
    if predictions.shape != truth.shape or candidates.shape[:2] != truth.shape:
        raise RuntimeError("frozen prediction geometry differs from truth")
    scored = scored_mask(mask)
    expected = truth[scored]
    predicted = predictions[scored]
    if predicted.lt(0).any().item():
        raise RuntimeError("fixed no-abstention policy emitted an invalid token")
    correct_mask = predicted.eq(expected)
    correct = int(correct_mask.sum().item())
    total = int(correct_mask.numel())
    k = int(entry["policy"]["spec"]["schedule"][-1])
    included = candidates[scored, :k].eq(expected[:, None]).any(dim=1)
    included_count = int(included.sum().item())
    conditional_correct = int((correct_mask & included).sum().item())
    exact, per_record = exact_input_summary(
        predictions=predictions,
        truth=truth,
        attention_mask=mask,
        tokenizer=tokenizer,
        source_texts=source_texts,
    )
    for index, row in enumerate(per_record):
        row["record_id"] = record_ids[index]
    cost = evidence["costs"][condition_id]["policies"][policy_id]
    return {
        "condition_id": condition_id,
        "proposer_id": proposer_id,
        "method_id": f"{proposer_id}__{entry['label']}",
        "policy_id": policy_id,
        "label": entry["label"],
        "policy": entry["policy"],
        "metrics": {
            "correct_tokens": correct,
            "scored_tokens": total,
            "token_accuracy": correct / total,
            **exact.as_dict(),
            "candidate_recall_hits": included_count,
            "candidate_recall": included_count / total,
            "conditional_selector_correct": conditional_correct,
            "conditional_selector_accuracy": (
                conditional_correct / included_count
                if included_count
                else None
            ),
            "coverage": 1.0,
        },
        "length_stratified": length_stratified_summary(
            per_record,
            bins=LENGTH_BINS if truth.shape[1] == 128 else ((40, 40),),
        ),
        "per_record": per_record,
        "cost": cost,
    }


def command_score(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve(strict=True)
    plan = load_json(args.plan)
    validate_plan(plan, args.selection_key)
    receipt = load_json(args.freeze_receipt)
    if (
        receipt.get("status") != "ALL_24_CELLS_FROZEN_BEFORE_HEAVY_TRUTH"
        or receipt.get("heavy_truth_opened") is not False
        or receipt.get("selection_key_opened") is not False
        or receipt.get("required_cells") != 24
    ):
        raise RuntimeError("R3 freeze receipt is invalid")
    require_file_record(receipt["plan"], args.plan, "plan")
    frozen = prediction_entries(receipt)
    if len(frozen) != 8:
        raise RuntimeError("frozen condition/proposer matrix is incomplete")

    commitment = load_json(args.selection_commitment)
    require_file_record(
        commitment["sanitized_config"],
        Path(commitment["sanitized_config"]["resolved_path"]),
        "heavy config",
    )
    require_file_record(
        commitment["sanitized_observations"],
        Path(commitment["sanitized_observations"]["resolved_path"]),
        "heavy observations",
    )
    key = args.selection_key.read_bytes()
    (
        heavy_truth,
        heavy_mask,
        heavy_ids,
        source_texts,
        mapping_rows,
    ) = load_heavy_truth(args.heavy_truth)
    mapping = [
        {
            "opaque_id": row["opaque_id"],
            "dataset_index": row["dataset_index"],
            "source_sha256": row["source_sha256"],
            "token_length": len(row["token_ids"]),
            "length_bin": row["length_bin"],
        }
        for row in mapping_rows
    ]
    observed_hmac = hmac.new(
        key, canonical_mapping_bytes(mapping), hashlib.sha256
    ).hexdigest()
    if observed_hmac != commitment["selection_hmac_sha256"]:
        raise RuntimeError("heavy selection reveal does not match commitment")
    reveal = {
        "schema": "token-reconstruction.trr0002-owner-r3-selection-reveal.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "REVEALED_AFTER_ALL_24_CELLS_FROZEN",
        "revealed_utc": utc_now(),
        "freeze_receipt": file_record(args.freeze_receipt),
        "selection_key_hex": key.hex(),
        "selection_key_sha256": hashlib.sha256(key).hexdigest(),
        "selection_hmac_sha256": observed_hmac,
        "mapping": mapping,
        "source_plaintext_disclosed": False,
    }
    write_json_exclusive(args.selection_reveal, reveal)

    clean_truth, clean_mask, clean_ids = load_clean_truth(
        args.clean_truth_jsonl
    )
    (
        finance_observations,
        finance_mask,
        finance_positions,
        finance_config,
        source300,
        captures,
    ) = historical_observations(args.historical_root.resolve(strict=True))
    del finance_observations, finance_positions
    tokenizer = source300.load_tokenizer(finance_config)
    finance_rows = source300.reconstruct_rows(
        captures, tokenizer, finance_config
    )
    finance_truth = torch.stack([row.truth_ids for row in finance_rows]).long()
    finance_ids = [
        f"source300-row-{row.row_index:03d}" for row in finance_rows
    ]
    truth_by_condition = {
        "clean_pile_lora_cut4": (
            clean_truth,
            clean_mask,
            clean_ids,
            None,
        ),
        "historical_finance_cut4": (
            finance_truth,
            finance_mask,
            finance_ids,
            None,
        ),
        "grandmaster_public_base_cut4": (
            heavy_truth,
            heavy_mask,
            heavy_ids,
            source_texts,
        ),
        "grandmaster_vikhr_heavy_cut4": (
            heavy_truth,
            heavy_mask,
            heavy_ids,
            source_texts,
        ),
    }

    cells: list[dict[str, Any]] = []
    for condition_id in CONDITION_IDS:
        truth, mask, ids, sources = truth_by_condition[condition_id]
        for proposer_id in PROPOSER_IDS:
            state, evidence = frozen[(condition_id, proposer_id)]
            for entry in POLICY_ENTRIES:
                cells.append(
                    score_cell(
                        condition_id=condition_id,
                        proposer_id=proposer_id,
                        entry=entry,
                        state=state,
                        evidence=evidence,
                        truth=truth,
                        mask=mask,
                        record_ids=ids,
                        tokenizer=tokenizer,
                        source_texts=sources,
                    )
                )
    if len(cells) != 24:
        raise RuntimeError("scored matrix is incomplete")
    index = {
        (cell["condition_id"], cell["proposer_id"], cell["label"]): cell
        for cell in cells
    }
    comparisons: dict[str, Any] = {
        "identity_vs_alpaca": {},
        "heavy_target_shift": {},
    }
    for condition_id in CONDITION_IDS:
        comparisons["identity_vs_alpaca"][condition_id] = {}
        for entry in POLICY_ENTRIES:
            strict = index[
                (condition_id, "checkpoint_identity", entry["label"])
            ]
            control = index[
                (condition_id, "alpaca_affine_control", entry["label"])
            ]
            comparisons["identity_vs_alpaca"][condition_id][entry["label"]] = {
                "token_accuracy_difference": (
                    strict["metrics"]["token_accuracy"]
                    - control["metrics"]["token_accuracy"]
                ),
                "correct_token_difference": (
                    strict["metrics"]["correct_tokens"]
                    - control["metrics"]["correct_tokens"]
                ),
                "exact_input_difference": (
                    strict["metrics"]["exact_token_records"]
                    - control["metrics"]["exact_token_records"]
                ),
                "runtime_ratio": (
                    strict["cost"]["method_compute_seconds"]
                    / control["cost"]["method_compute_seconds"]
                ),
            }
    for proposer_id in PROPOSER_IDS:
        comparisons["heavy_target_shift"][proposer_id] = {}
        for entry in POLICY_ENTRIES:
            matched = index[
                (
                    "grandmaster_public_base_cut4",
                    proposer_id,
                    entry["label"],
                )
            ]
            heavy = index[
                (
                    "grandmaster_vikhr_heavy_cut4",
                    proposer_id,
                    entry["label"],
                )
            ]
            comparisons["heavy_target_shift"][proposer_id][entry["label"]] = {
                "heavy_minus_matched_token_accuracy": (
                    heavy["metrics"]["token_accuracy"]
                    - matched["metrics"]["token_accuracy"]
                ),
                "heavy_minus_matched_correct_tokens": (
                    heavy["metrics"]["correct_tokens"]
                    - matched["metrics"]["correct_tokens"]
                ),
                "heavy_minus_matched_exact_inputs": (
                    heavy["metrics"]["exact_token_records"]
                    - matched["metrics"]["exact_token_records"]
                ),
            }

    roundtrip_truth_exact = sum(
        tokenizer.decode(
            row["token_ids"][1:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        == row["source_text"]
        for row in mapping_rows
    )
    preparation = load_json(args.heavy_preparation)
    result = {
        "schema": (
            "token-reconstruction.trr0002-owner-r3-strict-surrogate-result.v1"
        ),
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": (
            "COMPLETE_24_CELL_STRICT_SURROGATE_AND_HEAVY_TARGET_MATRIX"
        ),
        "ended_utc": utc_now(),
        "scoring_commit": git_head(root),
        "command": command_record(),
        "plan": file_record(args.plan),
        "freeze_receipt": file_record(args.freeze_receipt),
        "selection_commitment": file_record(args.selection_commitment),
        "selection_reveal": file_record(args.selection_reveal),
        "truth_opening": {
            "heavy_truth_loaded_after_freeze_verification": True,
            "selection_hmac_verified": True,
            "canonical_truth_already_open_retrospective": True,
        },
        "lineage": plan["lineage"],
        "heavy_target_weight_drift": preparation["model_weight_drift"],
        "heavy_panel": {
            "records": 64,
            "scored_tokens": int(heavy_mask[:, 1:].sum().item()),
            "truth_tokenizer_roundtrip_exact_source_strings": (
                roundtrip_truth_exact
            ),
            "lengths": preparation["lengths"],
        },
        "matrix": {
            "expected_cells": 24,
            "completed_cells": len(cells),
            "cells": cells,
        },
        "comparisons": comparisons,
        "claim_limits": [
            "The two canonical conditions are retrospective.",
            "The heavy target uses one public SFT derivative and one fixed test panel.",
            "No result is pooled across conditions.",
            "The Alpaca lens remains a control, not part of the strict method.",
        ],
    }
    write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cells": len(cells),
                "heavy_scored_tokens": result["heavy_panel"]["scored_tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    plan = load_json(args.plan)
    validate_plan(plan)
    receipt = load_json(args.freeze_receipt)
    commitment = load_json(args.selection_commitment)
    reveal = load_json(args.selection_reveal)
    result = load_json(args.result)
    checks = {
        "plan_status": plan["status"] == "FROZEN_BEFORE_R3_TARGET_OBSERVATIONS",
        "matrix_size": (
            plan.get("matrix", {}).get("required_cells") == 24
            and len(plan.get("matrix", {}).get("cells", [])) == 24
        ),
        "freeze_status": (
            receipt.get("status") == "ALL_24_CELLS_FROZEN_BEFORE_HEAVY_TRUTH"
        ),
        "freeze_entries": len(receipt.get("prediction_entries", [])) == 6,
        "common_record_batch_size_four": (
            receipt.get("operational_deviation", {}).get(
                "final_common_record_batch_size"
            ) == 4
            and receipt.get("operational_deviation", {}).get(
                "entire_matrix_rerun_consistently"
            ) is True
        ),
        "selection_commitment_matches": (
            commitment.get("selection_key_sha256")
            == reveal.get("selection_key_sha256")
        ),
        "selection_hmac_matches": (
            commitment.get("selection_hmac_sha256")
            == reveal.get("selection_hmac_sha256")
        ),
        "result_complete": (
            result.get("status")
            == "COMPLETE_24_CELL_STRICT_SURROGATE_AND_HEAVY_TARGET_MATRIX"
        ),
        "result_cells": len(result.get("matrix", {}).get("cells", [])) == 24,
        "all_cells_have_full_coverage": all(
            cell.get("metrics", {}).get("coverage") == 1.0
            for cell in result.get("matrix", {}).get("cells", [])
        ),
        "all_cells_have_exact_input_metrics": all(
            "exact_token_records" in cell.get("metrics", {})
            and "exact_decoded_text_records" in cell.get("metrics", {})
            for cell in result.get("matrix", {}).get("cells", [])
        ),
        "all_cells_have_cost_metrics": all(
            cell.get("cost", {}).get("candidate_simulations", 0) > 0
            and cell.get("cost", {}).get("method_compute_seconds", 0) > 0
            for cell in result.get("matrix", {}).get("cells", [])
        ),
        "heavy_target_weight_drift_positive": (
            result.get("heavy_target_weight_drift", {}).get("changed_tensors", 0)
            > 0
        ),
        "heavy_tokenizer_exactly_shared": (
            result["lineage"]["base"]["artifacts"]["tokenizer.json"]["sha256"]
            == result["lineage"]["heavy_target"]["artifacts"]["tokenizer.json"][
                "sha256"
            ]
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"validation failed: {failed}")

    artifacts = {}
    for name, path in {
        "plan": args.plan,
        "freeze_receipt": args.freeze_receipt,
        "selection_commitment": args.selection_commitment,
        "selection_reveal": args.selection_reveal,
        "result": args.result,
    }.items():
        artifacts[name] = file_record(path)

    payload = {
        "schema_version": "trr0002-r3-validation-v1",
        "status": "PASS",
        "checks": checks,
        "artifacts": artifacts,
        "validated_at_utc": utc_now(),
    }
    write_json_exclusive(args.output, payload)
    print(json.dumps({"status": "PASS", "checks": len(checks)}, sort_keys=True))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    commands = {
        "preregister": command_preregister,
        "prepare-heavy": command_prepare_heavy,
        "prepare-canonical": command_prepare_canonical,
        "predict": command_predict,
        "freeze": command_freeze,
        "score": command_score,
        "validate": command_validate,
    }
    return commands[args.command_name](args)


if __name__ == "__main__":
    raise SystemExit(main())
