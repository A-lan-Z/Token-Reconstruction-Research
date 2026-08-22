#!/usr/bin/env python3
"""Evaluator-only preparation for the frozen TRR-0001 blind reconstruction."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
from pathlib import Path
from typing import Any

from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from token_reconstruction.experiment_runtime import (
    CONDITIONS,
    CUT_DEPTHS,
    MODEL_ID,
    MODEL_REVISION,
    PhaseTimer,
    command_record,
    file_record,
    load_json,
    load_resources,
    peak_memory,
    records_for_split,
    require_create_only_directory,
    require_plan,
    seed_everything,
    utc_now,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from token_reconstruction.inverse import (
    InverseTrainingConfig,
    save_inverse,
    train_inverse,
)
from token_reconstruction.target_update import (
    TargetLoRAConfig,
    install_target_lora,
    save_target_lora,
    set_target_lora_enabled,
    target_lora_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def token_tensor(records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    value = torch.tensor([row["token_ids"] for row in records], dtype=torch.long)
    if value.shape[1] != 40:
        raise RuntimeError("record sequence length changed")
    return value.to(device)


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan)
    require_plan(plan)
    root = args.output_root.resolve()
    require_create_only_directory(root)
    public_root = root / "reconstructor_public"
    private_root = root / "evaluator_private"
    observations_root = public_root / "observations"
    inverse_root = public_root / "inverses"
    for directory in (public_root, private_root, observations_root, inverse_root):
        directory.mkdir()

    seed_everything(1729)
    torch.cuda.reset_peak_memory_stats()
    timer = PhaseTimer()
    started_utc = utc_now()

    with timer.measure("load_pinned_model_tokenizer_dataset"):
        tokenizer, dataset, model = load_resources()
    device = next(model.parameters()).device
    splits: dict[str, list[dict[str, Any]]] = {}
    with timer.measure("materialize_preregistered_record_tokens"):
        for split in (
            "target_update_train",
            "inverse_train",
            "development",
            "blind_evaluation",
        ):
            splits[split] = records_for_split(
                plan, split, tokenizer=tokenizer, dataset=dataset
            )

    model.requires_grad_(False)
    lora_config = TargetLoRAConfig()
    installed = install_target_lora(model, lora_config)
    parameters = target_lora_parameters(installed.values())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.0)
    target_records = splits["target_update_train"]
    target_losses: list[float] = []
    target_gradient_norms: list[float] = []
    model.train()
    with timer.measure("train_evaluator_only_target_lora"):
        for step in range(40):
            batch = [
                target_records[(step * 8 + offset) % len(target_records)]
                for offset in range(8)
            ]
            input_ids = token_tensor(batch, device)
            output = model(input_ids=input_ids, use_cache=False)
            logits = output.logits[:, :-1].float()
            labels = input_ids[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError("target-update loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            target_losses.append(float(loss.detach().cpu()))
            target_gradient_norms.append(float(gradient_norm.detach().cpu()))
    model.eval()
    target_path = private_root / "target_lora.safetensors"
    save_target_lora(installed, target_path)

    set_target_lora_enabled(installed.values(), False)
    inverse_records = splits["inverse_train"]
    public_activations: dict[int, list[torch.Tensor]] = {4: [], 8: []}
    public_targets: list[torch.Tensor] = []
    with timer.measure("collect_public_inverse_training_pairs"):
        with torch.inference_mode():
            for start in range(0, len(inverse_records), 8):
                input_ids = token_tensor(inverse_records[start : start + 8], device)
                output = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                public_targets.append(
                    model.get_input_embeddings()(input_ids)
                    .detach()
                    .reshape(-1, model.config.hidden_size)
                    .to(device="cpu", dtype=torch.float16)
                )
                for cut in (4, 8):
                    public_activations[cut].append(
                        output.hidden_states[cut]
                        .detach()
                        .reshape(-1, model.config.hidden_size)
                        .to(device="cpu", dtype=torch.float16)
                    )
                del output, input_ids
    target_embeddings = torch.cat(public_targets, dim=0)
    inverse_training: dict[str, Any] = {}
    for cut in (4, 8):
        activations = torch.cat(public_activations[cut], dim=0)
        with timer.measure(f"train_public_inverse_cut_{cut}"):
            inverse, evidence = train_inverse(
                activations,
                target_embeddings,
                config=InverseTrainingConfig(),
                device=device,
            )
        inverse_path = inverse_root / f"cut{cut}.safetensors"
        save_inverse(inverse, inverse_path, cut_depth=cut)
        evidence["artifact"] = file_record(inverse_path, root=root)
        inverse_training[str(cut)] = evidence
        del inverse, activations
        gc.collect()
        torch.cuda.empty_cache()

    frequency = Counter()
    for split in ("target_update_train", "inverse_train", "development"):
        for row in splits[split]:
            frequency.update(row["token_ids"][1:])
    frequency_path = public_root / "auxiliary_frequency_counts.json"
    write_json_exclusive(
        frequency_path,
        {
            "schema": "token-reconstruction.auxiliary-token-frequency.v1",
            "splits": ["target_update_train", "inverse_train", "development"],
            "scored_token_examples": int(sum(frequency.values())),
            "counts": {str(key): value for key, value in sorted(frequency.items())},
        },
    )

    blind = splits["blind_evaluation"]
    blind_ids = token_tensor(blind, device)
    observation_entries: list[dict[str, Any]] = []
    with timer.measure("generate_blind_boundary_observations"):
        for condition in CONDITIONS:
            set_target_lora_enabled(
                installed.values(), condition == "unavailable_target_lora"
            )
            collected: dict[int, list[torch.Tensor]] = {cut: [] for cut in CUT_DEPTHS}
            with torch.inference_mode():
                for start in range(0, len(blind), 8):
                    input_ids = blind_ids[start : start + 8]
                    output = model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    for cut in CUT_DEPTHS:
                        collected[cut].append(
                            output.hidden_states[cut]
                            .detach()
                            .to(device="cpu", dtype=torch.bfloat16)
                        )
                    del output
            for cut in CUT_DEPTHS:
                path = observations_root / f"{condition}_cut{cut}.safetensors"
                if path.exists() or path.is_symlink():
                    raise RuntimeError(f"observation already exists: {path}")
                tensor = torch.cat(collected[cut], dim=0).contiguous()
                save_file(
                    {"activations": tensor},
                    path,
                    metadata={
                        "schema": "token-reconstruction.boundary-observation.v1",
                        "condition": condition,
                        "cut_depth": str(cut),
                        "source_truth_included": "false",
                    },
                )
                observation_entries.append(
                    {
                        "condition": condition,
                        "cut_depth": cut,
                        "path": path.relative_to(public_root).as_posix(),
                        "tensor_key": "activations",
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                        "artifact": file_record(path, root=root),
                    }
                )

    observation_index_path = public_root / "observation_index.json"
    write_json_exclusive(
        observation_index_path,
        {
            "schema": "token-reconstruction.observation-index.v1",
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "declared_bos_token_id": 128000,
            "records": [
                {
                    "record_id": row["record_id"],
                    "dataset_index": row["dataset_index"],
                    "text_sha256": row["text_sha256"],
                }
                for row in blind
            ],
            "entries": observation_entries,
            "source_tokens_or_text_included": False,
        },
    )

    truth_path = private_root / "blind_truth.jsonl"
    write_jsonl_exclusive(
        truth_path,
        (
            {
                "record_id": row["record_id"],
                "dataset_index": row["dataset_index"],
                "text_sha256": row["text_sha256"],
                "token_ids": row["token_ids"],
            }
            for row in blind
        ),
    )

    ended_utc = utc_now()
    evidence_path = private_root / "evaluator_evidence.json"
    evidence = {
        "schema": "token-reconstruction.trr0001-evaluator-evidence.v1",
        "task_id": "TRR-0001",
        "command": command_record(),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "exit_status": 0,
        "phases": timer.records,
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
        },
        "target_update": {
            "config": {
                "layers": list(lora_config.layers),
                "modules": list(lora_config.modules),
                "rank": lora_config.rank,
                "alpha": lora_config.alpha,
                "steps": 40,
                "batch_records": 8,
                "learning_rate": 0.001,
                "seed": lora_config.seed,
            },
            "initial_loss": target_losses[0],
            "final_loss": target_losses[-1],
            "minimum_loss": min(target_losses),
            "losses": target_losses,
            "gradient_norms": target_gradient_norms,
            "trainable_parameters": sum(value.numel() for value in parameters),
            "artifact": file_record(target_path, root=root),
        },
        "inverse_training": inverse_training,
        "observations": {
            "index": file_record(observation_index_path, root=root),
            "entries": [entry["artifact"] for entry in observation_entries],
            "records": len(blind),
            "conditions": list(CONDITIONS),
            "cuts": list(CUT_DEPTHS),
        },
        "frequency_counts": file_record(frequency_path, root=root),
        "truth_sidecar": file_record(truth_path, root=root),
        "peak_memory": peak_memory(),
    }
    write_json_exclusive(evidence_path, evidence)
    print(
        {
            "status": "prepared",
            "records": len(blind),
            "scored_tokens_per_condition_cut": len(blind) * 39,
            "observation_index": str(observation_index_path),
            "truth_sidecar": str(truth_path),
            "evidence": str(evidence_path),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
