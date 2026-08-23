#!/usr/bin/env python3
"""Evaluator-only preparation of the owner-R1 fresh blind interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM

from token_reconstruction.a1a2_configuration_search import resolved_policy_from_dict
from token_reconstruction.blind_commitment import commitment_digest, require_exact_keys
from token_reconstruction.experiment_runtime import (
    BOS_TOKEN_ID,
    MODEL_ID,
    MODEL_REVISION,
    PhaseTimer,
    command_record,
    file_record,
    load_json,
    peak_memory,
    require_create_only_directory,
    seed_everything,
    sha256_file,
    utc_now,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from token_reconstruction.target_update import (
    TargetLoRAConfig,
    install_target_lora,
    load_target_lora,
)


METHOD_ID = "a1_a2_exhaustive_configuration_winner"
PUBLIC_SCHEMA = "token-reconstruction.trr0002-owner-r1-selection-commitment.v1"
PRIVATE_SCHEMA = "token-reconstruction.trr0002-owner-r1-private-selection.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-commitment", type=Path, required=True)
    parser.add_argument("--private-selection", type=Path, required=True)
    parser.add_argument("--target-lora", type=Path, required=True)
    parser.add_argument("--public-lens", type=Path, required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preregistration-commit", required=True)
    return parser.parse_args()


def copy_exclusive(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"required retained state is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def private_records(private: dict[str, Any], public: dict[str, Any]) -> list[dict[str, Any]]:
    require_exact_keys(
        private,
        {"schema", "task_id", "revision_id", "created_utc", "selection_key_hex", "records"},
        label="owner-R1 private selection",
    )
    if private["schema"] != PRIVATE_SCHEMA or private["task_id"] != "TRR-0002":
        raise RuntimeError("private selection schema changed")
    try:
        key = bytes.fromhex(str(private["selection_key_hex"]))
    except ValueError as exc:
        raise RuntimeError("private selection key is invalid") from exc
    records = private["records"]
    expected_ids = [f"blind-r1-{position:06d}" for position in range(1, 65)]
    if len(key) != 32 or not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("private selection geometry changed")
    if [row.get("record_id") for row in records] != expected_ids:
        raise RuntimeError("owner-R1 opaque record order changed")
    if commitment_digest(key, records) != public["commitment"]:
        raise RuntimeError("private selection does not match its public commitment")
    for row in records:
        require_exact_keys(
            row,
            {"record_id", "dataset_index", "text_sha256", "token_ids"},
            label="owner-R1 private record",
        )
        if len(row["token_ids"]) != 40 or int(row["token_ids"][0]) != BOS_TOKEN_ID:
            raise RuntimeError("private truth token geometry changed")
    return records


def artifact_entry(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_model() -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("blind evaluator requires CUDA")
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
    return model


def main() -> int:
    args = parse_args()
    if len(args.preregistration_commit) != 40:
        raise RuntimeError("full blind preregistration commit is required")
    plan = load_json(args.plan)
    if (
        plan.get("schema")
        != "token-reconstruction.trr0002-owner-r1-blind-preregistration.v1"
        or plan.get("status")
        != "COMMITTED_BEFORE_OWNER_R1_FRESH_BLIND_OBSERVATIONS"
        or plan.get("truth_opened") is not False
        or plan.get("method_id") != METHOD_ID
    ):
        raise RuntimeError("owner-R1 blind preregistration changed")
    public = load_json(args.public_commitment)
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or public.get("task_id") != "TRR-0002"
        or public.get("record_count") != 64
        or public.get("source_identity_disclosed") is not False
        or public.get("selection_key_disclosed") is not False
    ):
        raise RuntimeError("owner-R1 public commitment changed")
    records = private_records(load_json(args.private_selection), public)
    for argument, key in (
        (args.target_lora, "target_lora"),
        (args.public_lens, "public_lens"),
    ):
        if sha256_file(argument) != plan["retained_state"][key]["sha256"]:
            raise RuntimeError(f"retained state changed: {key}")
    if sha256_file(args.winner) != plan["winner"]["sha256"]:
        raise RuntimeError("frozen winner hash changed")
    winner = load_json(args.winner)
    policy = resolved_policy_from_dict(winner["policy"])
    if policy.policy_id != plan["policy_id"] or policy.serialized() != plan["policy"]:
        raise RuntimeError("frozen policy differs from preregistration")

    root = args.output_root.resolve()
    require_create_only_directory(root)
    public_root = root / "reconstructor_input"
    private_root = root / "evaluator_private"
    public_root.mkdir()
    private_root.mkdir()
    started_utc = utc_now()
    seed_everything(20260826)
    torch.cuda.reset_peak_memory_stats()
    timer = PhaseTimer()
    with timer.measure("load_pinned_model_and_unavailable_update"):
        model = load_model()
        installed = install_target_lora(model, TargetLoRAConfig())
        load_target_lora(installed, args.target_lora)
        model.requires_grad_(False)
        model.eval()
    tokens = torch.tensor(
        [[int(value) for value in row["token_ids"]] for row in records],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    if tuple(tokens.shape) != (64, 40):
        raise RuntimeError("fresh blind token tensor geometry changed")
    collected: list[torch.Tensor] = []
    with timer.measure("generate_unavailable_lora_cut4_observations"):
        with torch.inference_mode():
            for start in range(0, 64, 8):
                output = model(
                    input_ids=tokens[start : start + 8],
                    output_hidden_states=True,
                    use_cache=False,
                )
                collected.append(
                    output.hidden_states[4].detach().to(device="cpu", dtype=torch.bfloat16)
                )
                del output
    observations = torch.cat(collected, dim=0).contiguous()
    if tuple(observations.shape) != (64, 40, 2048):
        raise RuntimeError("fresh blind observation geometry changed")
    observation_path = public_root / "unavailable_target_lora_cut4.safetensors"
    save_file(
        {"activations": observations},
        observation_path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r1-blind-observation.v1",
            "condition": "unavailable_target_lora",
            "cut_depth": "4",
            "opaque_records": "true",
            "source_truth_included": "false",
        },
    )

    lens_destination = public_root / "public_a1_lens.pt"
    winner_destination = public_root / "frozen_winner.json"
    with timer.measure("copy_exact_public_method_state"):
        copy_exclusive(args.public_lens, lens_destination)
        copy_exclusive(args.winner, winner_destination)
    observation_index_path = public_root / "observation_index.json"
    write_json_exclusive(
        observation_index_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-blind-observation-index.v1",
            "records": [{"record_id": row["record_id"]} for row in records],
            "observation": artifact_entry(observation_path, relative_to=public_root),
            "source_material_included": False,
        },
    )
    config_path = public_root / "sanitized_config.json"
    config = {
        "schema": "token-reconstruction.trr0002-owner-r1-blind-sanitized-config.v1",
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R1",
        "method_id": METHOD_ID,
        "policy_id": policy.policy_id,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "prefix_layers": [0, 1, 2, 3],
        },
        "record_order": [row["record_id"] for row in records],
        "geometry": {
            "records": 64,
            "sequence_tokens": 40,
            "scored_tokens_per_record": 39,
            "hidden_size": 2048,
            "cut_depth": 4,
        },
        "observation_index": artifact_entry(observation_index_path, relative_to=public_root),
        "observation": artifact_entry(observation_path, relative_to=public_root),
        "public_lens": artifact_entry(lens_destination, relative_to=public_root),
        "winner": artifact_entry(winner_destination, relative_to=public_root),
        "execution": {
            "seed": 20260826,
            "record_batch_size": 8,
            "maximum_candidate_budget": max(policy.spec.schedule),
            "terminal_action": policy.spec.terminal_action,
            "target_prefix_calls": 0,
        },
        "truth_or_source_inputs": 0,
        "access_contract": "minimal read-only namespace; no workspace, dataset, unavailable update, truth, private selection, historical source, canonical truth, or network",
    }
    write_json_exclusive(config_path, config)

    truth_path = private_root / "blind_truth.jsonl"
    write_jsonl_exclusive(
        truth_path,
        ({"record_id": row["record_id"], "token_ids": row["token_ids"]} for row in records),
    )
    evidence_path = private_root / "evaluator_evidence.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-blind-evaluator-evidence.v1",
            "task_id": "TRR-0002",
            "status": "OWNER_R1_FRESH_BLIND_INTERFACE_PREPARED",
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "command": command_record(),
            "exit_status": 0,
            "preregistration_commit": args.preregistration_commit,
            "plan": file_record(args.plan),
            "public_commitment": file_record(args.public_commitment),
            "retained_state": {
                "target_lora": file_record(args.target_lora),
                "public_lens": file_record(args.public_lens),
                "winner": file_record(args.winner),
                "fresh_training_steps": 0,
                "fresh_adaptation_steps": 0,
            },
            "public_interface": {
                "config": file_record(config_path, root=root),
                "observation_index": file_record(observation_index_path, root=root),
                "observation": file_record(observation_path, root=root),
                "source_identity_fields": 0,
                "source_token_or_text_fields": 0,
                "opaque_records": 64,
            },
            "private_outputs": {"truth": file_record(truth_path, root=root)},
            "phases": timer.records,
            "peak_memory": peak_memory(),
        },
    )
    for path in sorted(public_root.rglob("*")):
        if path.is_file():
            path.chmod(0o444)
    print(
        json.dumps(
            {
                "status": "OWNER_R1_FRESH_BLIND_INTERFACE_PREPARED",
                "records": 64,
                "scored_tokens": 2496,
                "policy_id": policy.policy_id,
                "public_config": str(config_path),
                "source_identity_fields": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
