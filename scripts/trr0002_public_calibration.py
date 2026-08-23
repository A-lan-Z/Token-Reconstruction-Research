#!/usr/bin/env python3
"""Generate disjoint public-development updates and cut-4 observations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from safetensors.torch import save_file
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_reconstruction.experiment_runtime import (
    BOS_TOKEN_ID,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PhaseTimer,
    command_record,
    file_record,
    load_json,
    peak_memory,
    seed_everything,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.target_update import (
    TargetLoRAConfig,
    install_target_lora,
    target_lora_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_records(
    *,
    source_plan: dict[str, Any],
    split: str,
    tokenizer: Any,
    dataset: Any,
) -> list[dict[str, Any]]:
    declared = source_plan["data"]["selection"]["splits"][split]["records"]
    output: list[dict[str, Any]] = []
    for row in declared:
        index = int(row["index"])
        text = str(dataset[index]["text"])
        if sha256_text(text) != row["text_sha256"]:
            raise RuntimeError(f"public source text hash changed at index {index}")
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(encoded) < 39:
            raise RuntimeError("preregistered public record is no longer eligible")
        tokens = [BOS_TOKEN_ID, *[int(value) for value in encoded[:39]]]
        output.append(
            {
                "record_id": row["record_id"],
                "dataset_index": index,
                "text_sha256": row["text_sha256"],
                "token_ids": tokens,
            }
        )
    return output


def load_model() -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("public calibration generation requires CUDA")
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
    if model.config.hidden_size != 2048 or model.config.vocab_size != 128256:
        raise RuntimeError("pinned public model geometry changed")
    model.requires_grad_(False)
    return model


def token_tensor(records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    value = torch.tensor([row["token_ids"] for row in records], dtype=torch.long)
    if tuple(value.shape)[1:] != (40,):
        raise RuntimeError("public calibration sequence geometry changed")
    return value.to(device)


@torch.inference_mode()
def capture_cut4(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
) -> torch.Tensor:
    device = next(model.parameters()).device
    tokens = token_tensor(records, device)
    collected: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(records), 8):
        output = model(
            input_ids=tokens[start : start + 8],
            output_hidden_states=True,
            use_cache=False,
        )
        collected.append(
            output.hidden_states[4].detach().to(device="cpu", dtype=torch.bfloat16)
        )
        del output
    result = torch.cat(collected, dim=0).contiguous()
    if tuple(result.shape) != (len(records), 40, 2048):
        raise RuntimeError("public calibration observation geometry changed")
    return result


def save_public_update(installed: dict[str, Any], path: Path, condition: str) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for name, module in installed.items():
        tensors[f"{name}.A"] = module.A.detach().cpu().contiguous()
        tensors[f"{name}.B"] = module.B.detach().cpu().contiguous()
    save_file(
        tensors,
        path,
        metadata={
            "schema": "token-reconstruction.trr0002-public-development-lora.v1",
            "access": "public-development-only",
            "condition": condition,
        },
    )


def train_update(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    condition: dict[str, Any],
) -> tuple[dict[str, Any], list[float], list[float]]:
    seed = int(condition["seed"])
    seed_everything(seed)
    config = TargetLoRAConfig(
        layers=tuple(int(value) for value in condition["layers"]),
        modules=tuple(str(value) for value in condition["modules"]),
        rank=int(condition["rank"]),
        alpha=float(condition["alpha"]),
        seed=seed,
    )
    installed = install_target_lora(model, config)
    parameters = target_lora_parameters(installed.values())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(condition["learning_rate"]),
        weight_decay=float(condition["weight_decay"]),
    )
    device = next(model.parameters()).device
    losses: list[float] = []
    gradient_norms: list[float] = []
    model.train()
    batch_size = int(condition["batch_records"])
    for step in range(int(condition["steps"])):
        batch = [
            records[(step * batch_size + offset) % len(records)]
            for offset in range(batch_size)
        ]
        input_ids = token_tensor(batch, device)
        output = model(input_ids=input_ids, use_cache=False)
        logits = output.logits[:, :-1].float()
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if not torch.isfinite(loss).item():
            raise RuntimeError("public development update loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(condition["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))
    model.eval()
    return installed, losses, gradient_norms


def training_records_for(
    condition_id: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(records) != 64:
        raise RuntimeError("public update training split changed")
    if condition_id == "public_lora_2601":
        return records[:32]
    if condition_id == "public_lora_2602":
        return records[32:]
    if condition_id == "public_lora_2603":
        return list(reversed(records))
    raise RuntimeError(f"unknown public development update: {condition_id}")


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan)
    source_plan = load_json(args.source_plan)
    if plan.get("schema") != "token-reconstruction.trr0002-calibration-preregistration.v1":
        raise RuntimeError("calibration preregistration changed")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError("public calibration output is create-only")
    args.output_root.mkdir(parents=True)
    observations_root = args.output_root / "observations"
    updates_root = args.output_root / "updates"
    observations_root.mkdir()
    updates_root.mkdir()

    started_utc = utc_now()
    timer = PhaseTimer()
    seed_everything(2599)
    torch.cuda.reset_peak_memory_stats()
    with timer.measure("load_public_tokenizer_and_dataset"):
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
        dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    with timer.measure("materialize_disjoint_public_records"):
        development = load_records(
            source_plan=source_plan,
            split="development",
            tokenizer=tokenizer,
            dataset=dataset,
        )
        update_train = load_records(
            source_plan=source_plan,
            split="target_update_train",
            tokenizer=tokenizer,
            dataset=dataset,
        )
    if {row["dataset_index"] for row in development} & {
        row["dataset_index"] for row in update_train
    }:
        raise RuntimeError("public development and update training overlap")

    truth_path = args.output_root / "truth.safetensors"
    save_file(
        {"token_ids": torch.tensor([row["token_ids"] for row in development], dtype=torch.int32)},
        truth_path,
        metadata={
            "schema": "token-reconstruction.trr0002-public-development-truth.v1",
            "access": "public-auxiliary",
        },
    )
    record_index_path = args.output_root / "records.json"
    write_json_exclusive(
        record_index_path,
        {
            "schema": "token-reconstruction.trr0002-public-development-records.v1",
            "development": development,
            "update_train": update_train,
            "disjoint": True,
        },
    )

    conditions = plan["public_development"]["conditions"]
    if [row["id"] for row in conditions] != [
        "public_base",
        "public_lora_2601",
        "public_lora_2602",
        "public_lora_2603",
    ]:
        raise RuntimeError("public development condition order changed")
    condition_evidence: list[dict[str, Any]] = []

    with timer.measure("load_base_and_capture_public_base"):
        model = load_model()
        base_observation = capture_cut4(model, development)
        base_path = observations_root / "public_base_cut4.safetensors"
        save_file(
            {"activations": base_observation},
            base_path,
            metadata={
                "schema": "token-reconstruction.trr0002-public-development-observation.v1",
                "condition": "public_base",
                "cut_depth": "4",
            },
        )
    condition_evidence.append(
        {
            "id": "public_base",
            "role": "threshold fitting",
            "observation": file_record(base_path),
            "update": None,
            "training": None,
        }
    )
    del base_observation, model
    gc.collect()
    torch.cuda.empty_cache()

    for condition in conditions[1:]:
        condition_id = str(condition["id"])
        train_records = training_records_for(condition_id, update_train)
        with timer.measure(f"load_train_capture_{condition_id}"):
            model = load_model()
            installed, losses, norms = train_update(
                model=model,
                records=train_records,
                condition=condition,
            )
            update_path = updates_root / f"{condition_id}.safetensors"
            save_public_update(installed, update_path, condition_id)
            observation = capture_cut4(model, development)
            observation_path = observations_root / f"{condition_id}_cut4.safetensors"
            save_file(
                {"activations": observation},
                observation_path,
                metadata={
                    "schema": "token-reconstruction.trr0002-public-development-observation.v1",
                    "condition": condition_id,
                    "cut_depth": "4",
                },
            )
        condition_evidence.append(
            {
                "id": condition_id,
                "role": condition["role"],
                "observation": file_record(observation_path),
                "update": file_record(update_path),
                "training": {
                    "config": condition,
                    "record_count": len(train_records),
                    "record_ids": [row["record_id"] for row in train_records],
                    "loss_first": losses[0],
                    "loss_last": losses[-1],
                    "loss_minimum": min(losses),
                    "losses": losses,
                    "gradient_norms": norms,
                },
            }
        )
        del observation, installed, model
        gc.collect()
        torch.cuda.empty_cache()

    evidence_path = args.output_root / "generation.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr0002-public-calibration-generation.v1",
            "task_id": "TRR-0002",
            "status": "PUBLIC_DEVELOPMENT_GENERATED",
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": command_record(),
            "exit_status": 0,
            "plan": file_record(args.plan),
            "source_plan": file_record(args.source_plan),
            "records": file_record(record_index_path),
            "truth": file_record(truth_path),
            "conditions": condition_evidence,
            "phases": timer.records,
            "peak_memory": peak_memory(),
            "canonical_evaluation_truth_inputs": 0,
            "canonical_evaluation_observation_inputs": 0,
            "target_lora_inputs": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "PUBLIC_DEVELOPMENT_GENERATED",
                "conditions": [row["id"] for row in condition_evidence],
                "development_records": len(development),
                "update_training_records": len(update_train),
                "output": str(evidence_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
