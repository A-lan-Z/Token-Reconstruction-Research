#!/usr/bin/env python3
"""Create the preregistered disjoint public Finance development family."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset
from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_reconstruction.experiment_runtime import (
    BOS_TOKEN_ID,
    MODEL_ID,
    MODEL_REVISION,
    PhaseTimer,
    command_record,
    file_record,
    peak_memory,
    seed_everything,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.target_update import (
    TargetLoRAConfig,
    install_target_lora,
    load_target_lora,
)


DATASET_ID = "Josephgflowers/Finance-Instruct-500k"
DATASET_FINGERPRINT = "4abbac8acaab4205"
SHUFFLE_SEED = 42
HISTORICAL_CURSOR_START = 38847
PUBLIC_CURSOR_START = 38978
RECORDS = 32
TOKENS = 128
DATE_STRING = "06 Aug 2026"
CONDITIONS = (
    "public_base",
    "public_lora_2601",
    "public_lora_2602",
    "public_lora_2603",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_fields(row: Mapping[str, Any]) -> tuple[str | None, str, str]:
    user = str(row.get("user") or "").strip()
    assistant = str(row.get("assistant") or "").strip()
    system_text = str(row.get("system") or "").strip()
    return system_text or None, user, assistant


def content_hash(system: str | None, user: str, assistant: str) -> str:
    payload = json.dumps(
        [system, user, assistant],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def template_ids(
    tokenizer: Any,
    *,
    system: str | None,
    user: str,
    assistant: str,
) -> list[int]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    output = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        date_string=DATE_STRING,
    )
    if hasattr(output, "keys") and "input_ids" in output:
        ids = output["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(value) for value in ids]
    return [int(value) for value in output]


def select_records(tokenizer: Any, dataset: Any) -> tuple[list[dict[str, Any]], int]:
    historical_hashes: set[str] = set()
    for index in range(HISTORICAL_CURSOR_START, PUBLIC_CURSOR_START):
        system, user, assistant = normalized_fields(dataset[index])
        if user and assistant:
            historical_hashes.add(content_hash(system, user, assistant))

    selected: list[dict[str, Any]] = []
    cursor = PUBLIC_CURSOR_START
    pad_token_id = int(tokenizer.pad_token_id)
    while len(selected) < RECORDS:
        if cursor >= len(dataset):
            raise RuntimeError("finance dataset exhausted")
        raw_index = cursor
        cursor += 1
        system, user, assistant = normalized_fields(dataset[raw_index])
        if not user or not assistant:
            continue
        row_hash = content_hash(system, user, assistant)
        if row_hash in historical_hashes:
            raise RuntimeError("public finance row duplicates historical cursor content")
        ids = template_ids(
            tokenizer,
            system=system,
            user=user,
            assistant=assistant,
        )[:TOKENS]
        if not ids or ids[0] != BOS_TOKEN_ID:
            raise RuntimeError("public finance chat template lost BOS")
        valid_tokens = len(ids)
        padded = [*ids, *([pad_token_id] * (TOKENS - valid_tokens))]
        mask = [1] * valid_tokens + [0] * (TOKENS - valid_tokens)
        positions = [max(0, sum(mask[: index + 1]) - 1) for index in range(TOKENS)]
        token_hash = sha256_bytes(
            torch.tensor(ids, dtype=torch.int32).numpy().tobytes()
        )
        selected.append(
            {
                "record_id": f"finance-public-{raw_index:06d}-{row_hash[:16]}",
                "raw_index": raw_index,
                "content_sha256": row_hash,
                "token_ids_sha256": token_hash,
                "valid_tokens": valid_tokens,
                "input_ids": padded,
                "attention_mask": mask,
                "position_ids": positions,
            }
        )
    if len({row["raw_index"] for row in selected}) != RECORDS:
        raise RuntimeError("public finance raw indices are not unique")
    if min(row["raw_index"] for row in selected) < PUBLIC_CURSOR_START:
        raise RuntimeError("public finance selection overlaps historical cursor")
    return selected, cursor


def tensors(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.tensor([row["input_ids"] for row in records], dtype=torch.long)
    mask = torch.tensor([row["attention_mask"] for row in records], dtype=torch.long)
    positions = torch.tensor([row["position_ids"] for row in records], dtype=torch.long)
    if ids.shape != (RECORDS, TOKENS) or mask.shape != ids.shape or positions.shape != ids.shape:
        raise RuntimeError("public finance tensor geometry changed")
    return ids, mask, positions


def load_model() -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("public finance observation generation requires CUDA")
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(torch.device("cuda"))
        .eval()
    )
    model.requires_grad_(False)
    if model.config.hidden_size != 2048 or model.config.vocab_size != 128256:
        raise RuntimeError("pinned public model geometry changed")
    return model


def install_condition(
    model: torch.nn.Module,
    condition: Mapping[str, Any],
    update_path: Path,
) -> None:
    config = TargetLoRAConfig(
        layers=tuple(int(value) for value in condition["layers"]),
        modules=tuple(str(value) for value in condition["modules"]),
        rank=int(condition["rank"]),
        alpha=float(condition["alpha"]),
        seed=int(condition["seed"]),
    )
    installed = install_target_lora(model, config)
    load_target_lora(installed, update_path)


@torch.inference_mode()
def capture_cut4(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    device = next(model.parameters()).device
    values: list[torch.Tensor] = []
    for start in range(0, input_ids.shape[0], 4):
        output = model(
            input_ids=input_ids[start : start + 4].to(device),
            attention_mask=attention_mask[start : start + 4].to(device),
            position_ids=position_ids[start : start + 4].to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        values.append(
            output.hidden_states[4].detach().to(device="cpu", dtype=torch.bfloat16)
        )
        del output
    result = torch.cat(values, dim=0).contiguous()
    if result.shape != (RECORDS, TOKENS, 2048):
        raise RuntimeError("public finance observation geometry changed")
    return result


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "token-reconstruction.trr0002-owner-r1-configuration-search-preregistration.v1":
        raise RuntimeError("configuration-search preregistration changed")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError("public finance output is create-only")
    args.output_root.mkdir(parents=True)
    observation_root = args.output_root / "observations"
    observation_root.mkdir()

    started_utc = utc_now()
    timer = PhaseTimer()
    seed_everything(20260823)
    torch.cuda.reset_peak_memory_stats()
    with timer.measure("load_public_tokenizer_and_finance_dataset"):
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
        dataset = load_dataset(DATASET_ID, split="train").shuffle(seed=SHUFFLE_SEED)
    if str(dataset._fingerprint) != DATASET_FINGERPRINT:
        raise RuntimeError("public finance dataset fingerprint changed")

    with timer.measure("select_and_hash_disjoint_finance_records"):
        records, cursor_end = select_records(tokenizer, dataset)
        input_ids, attention_mask, position_ids = tensors(records)

    truth_path = args.output_root / "truth.safetensors"
    save_file(
        {
            "token_ids": input_ids.to(torch.int32).contiguous(),
            "attention_mask": attention_mask.to(torch.uint8).contiguous(),
            "position_ids": position_ids.to(torch.int32).contiguous(),
        },
        truth_path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r1-public-finance-truth.v1",
            "access": "public-auxiliary",
            "historical_overlap": "none-by-raw-cursor-and-content-sha256",
        },
    )
    records_path = args.output_root / "records.json"
    write_json_exclusive(
        records_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-public-finance-records.v1",
            "dataset": DATASET_ID,
            "fingerprint": DATASET_FINGERPRINT,
            "shuffle_seed": SHUFFLE_SEED,
            "historical_cursor": [HISTORICAL_CURSOR_START, PUBLIC_CURSOR_START],
            "public_cursor": [PUBLIC_CURSOR_START, cursor_end],
            "chat_template_date": DATE_STRING,
            "records": records,
            "overlap": {
                "raw_index": 0,
                "normalized_content_sha256": 0,
            },
        },
    )

    condition_configs = {
        str(row["id"]): row for row in plan["public_data"].get("conditions", [])
    }
    # The exact LoRA hyperparameters live in the already-frozen calibration plan.
    calibration_plan = json.loads(
        Path("experiments/TRR-0002/calibration/preregistration/plan.json").read_text(
            encoding="utf-8"
        )
    )
    condition_configs = {
        str(row["id"]): row
        for row in calibration_plan["public_development"]["conditions"]
    }
    if tuple(condition_configs) != CONDITIONS:
        raise RuntimeError("public condition registry changed")

    condition_evidence: list[dict[str, Any]] = []
    for condition_id in CONDITIONS:
        with timer.measure(f"capture_{condition_id}"):
            model = load_model()
            update_record = None
            if condition_id != "public_base":
                update_path = args.public_root / "updates" / f"{condition_id}.safetensors"
                install_condition(model, condition_configs[condition_id], update_path)
                update_record = file_record(update_path)
            observation = capture_cut4(model, input_ids, attention_mask, position_ids)
            observation_path = observation_root / f"{condition_id}_cut4.safetensors"
            save_file(
                {"activations": observation},
                observation_path,
                metadata={
                    "schema": "token-reconstruction.trr0002-owner-r1-public-finance-observation.v1",
                    "condition": condition_id,
                    "cut_depth": "4",
                },
            )
        condition_evidence.append(
            {
                "id": condition_id,
                "config": condition_configs[condition_id],
                "update": update_record,
                "observation": file_record(observation_path),
            }
        )
        del observation, model
        gc.collect()
        torch.cuda.empty_cache()

    evidence_path = args.output_root / "generation.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-public-finance-generation.v1",
            "task_id": "TRR-0002",
            "revision_id": "TRR-0002-OWNER-REVISION-R1",
            "status": "PUBLIC_FINANCE_GENERATED",
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": command_record(),
            "exit_status": 0,
            "plan": file_record(args.plan),
            "calibration_plan": file_record(
                Path("experiments/TRR-0002/calibration/preregistration/plan.json")
            ),
            "source_public_calibration": file_record(args.public_root / "generation.json"),
            "records": file_record(records_path),
            "truth": file_record(truth_path),
            "conditions": condition_evidence,
            "phases": timer.records,
            "peak_memory": peak_memory(),
            "canonical_evaluation_observation_inputs": 0,
            "canonical_evaluation_truth_inputs": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "PUBLIC_FINANCE_GENERATED",
                "records": RECORDS,
                "cursor_end": cursor_end,
                "conditions": list(CONDITIONS),
                "output": str(evidence_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
