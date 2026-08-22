#!/usr/bin/env python3
"""Evaluator-only fresh observation generation for TRR-0001-R1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any

from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_reconstruction.blind_commitment import (
    OBSERVATION_INDEX_SCHEMA,
    PRIVATE_SELECTION_SCHEMA,
    SANITIZED_CONFIG_SCHEMA,
    commitment_digest,
    require_exact_keys,
    require_opaque_record_order,
    validate_observation_index,
    validate_public_commitment,
    validate_sanitized_config,
)
from token_reconstruction.experiment_runtime import (
    BOS_TOKEN_ID,
    CONDITIONS,
    CUT_DEPTHS,
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
    set_target_lora_enabled,
)


TARGET_LORA_SHA256 = "34d92f1e664236bfa1990b10148e8ad52c60b16e72ed0ff4c7eb7da8d15019f6"
INVERSE_SHA256 = {
    4: "9e2487f85057748130bf87b2aad0a883f3c36dfc052eefd83c0f5c35497a24e3",
    8: "ac8871f1fa0d40664c5d9d94343ef560832477aade3574978a1c6b572df01e80",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-commitment", type=Path, required=True)
    parser.add_argument("--private-selection", type=Path, required=True)
    parser.add_argument("--target-lora", type=Path, required=True)
    parser.add_argument("--inverse-directory", type=Path, required=True)
    parser.add_argument("--frequency-counts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preregistration-commit", required=True)
    return parser.parse_args()


def copy_exclusive(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"retained state is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def load_public_model_and_tokenizer() -> tuple[Any, torch.nn.Module]:
    if not torch.cuda.is_available():
        raise RuntimeError("TRR-0001-R1 evaluator requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    if tokenizer.bos_token_id != BOS_TOKEN_ID:
        raise RuntimeError("pinned tokenizer BOS changed")
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
    return tokenizer, model


def private_records(private: dict, public: dict) -> tuple[bytes, list[dict[str, Any]]]:
    require_exact_keys(
        private,
        {"schema", "created_utc", "selection_key_hex", "records"},
        label="private selection",
    )
    if private["schema"] != PRIVATE_SELECTION_SCHEMA:
        raise RuntimeError("private selection schema changed")
    try:
        key = bytes.fromhex(str(private["selection_key_hex"]))
    except ValueError as exc:
        raise RuntimeError("private selection key is invalid") from exc
    if len(key) != 32:
        raise RuntimeError("private selection key length changed")
    records = private["records"]
    require_opaque_record_order([row.get("record_id") for row in records])
    if commitment_digest(key, records) != public["commitment"]:
        raise RuntimeError("private selection does not match public commitment")
    for row in records:
        require_exact_keys(
            row,
            {"record_id", "dataset_index", "text_sha256", "token_ids"},
            label="private record",
        )
        if len(row["token_ids"]) != 40 or int(row["token_ids"][0]) != BOS_TOKEN_ID:
            raise RuntimeError("private truth token geometry changed")
    return key, records


def artifact_entry(path: Path, *, relative_to: Path, **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = parse_args()
    if len(args.preregistration_commit) != 40:
        raise RuntimeError("full preregistration commit is required")
    plan = load_json(args.plan)
    if (
        plan.get("schema") != "token-reconstruction.trr0001-r1.preregistration.v1"
        or plan.get("status") != "COMMITTED_BEFORE_CLEAN_CONFIRMATORY_OUTPUTS"
        or plan.get("truth_opened") is not False
    ):
        raise RuntimeError("R1 preregistration identity changed")
    public = load_json(args.public_commitment)
    validate_public_commitment(public)
    private = load_json(args.private_selection)
    _, records = private_records(private, public)

    if sha256_file(args.target_lora) != TARGET_LORA_SHA256:
        raise RuntimeError("retained target LoRA hash changed")
    for cut, expected in INVERSE_SHA256.items():
        if sha256_file(args.inverse_directory / f"cut{cut}.safetensors") != expected:
            raise RuntimeError(f"retained public inverse hash changed at cut {cut}")

    root = args.output_root.resolve()
    require_create_only_directory(root)
    public_root = root / "reconstructor_input"
    private_root = root / "evaluator_private"
    observations_root = public_root / "observations"
    inverse_root = public_root / "inverses"
    for directory in (public_root, private_root, observations_root, inverse_root):
        directory.mkdir()

    started_utc = utc_now()
    seed_everything(1729)
    torch.cuda.reset_peak_memory_stats()
    timer = PhaseTimer()
    with timer.measure("load_pinned_public_model_and_tokenizer"):
        _, model = load_public_model_and_tokenizer()
    with timer.measure("install_and_load_exact_retained_target_lora"):
        installed = install_target_lora(model, TargetLoRAConfig())
        load_target_lora(installed, args.target_lora)
        model.requires_grad_(False)
        model.eval()

    with timer.measure("copy_exact_public_inverse_states"):
        inverse_entries = []
        for cut in (4, 8):
            source = args.inverse_directory / f"cut{cut}.safetensors"
            destination = inverse_root / source.name
            copy_exclusive(source, destination)
            inverse_entries.append(
                artifact_entry(
                    destination,
                    relative_to=public_root,
                    cut_depth=cut,
                )
            )
        frequency_destination = private_root / "auxiliary_frequency_counts.json"
        copy_exclusive(args.frequency_counts, frequency_destination)

    token_ids = torch.tensor(
        [[int(value) for value in row["token_ids"]] for row in records],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    if tuple(token_ids.shape) != (64, 40):
        raise RuntimeError("fresh truth token tensor geometry changed")
    observation_entries: list[dict[str, Any]] = []
    with timer.measure("generate_fresh_boundary_observations"):
        for condition in CONDITIONS:
            set_target_lora_enabled(
                installed.values(), condition == "unavailable_target_lora"
            )
            collected: dict[int, list[torch.Tensor]] = {cut: [] for cut in CUT_DEPTHS}
            with torch.inference_mode():
                for start in range(0, 64, 8):
                    output = model(
                        input_ids=token_ids[start : start + 8],
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
                tensor = torch.cat(collected[cut], dim=0).contiguous()
                if tuple(tensor.shape) != (64, 40, 2048):
                    raise RuntimeError("fresh observation tensor geometry changed")
                path = observations_root / f"{condition}_cut{cut}.safetensors"
                save_file(
                    {"activations": tensor},
                    path,
                    metadata={
                        "schema": "token-reconstruction.trr0001-r1-boundary-observation.v1",
                        "condition": condition,
                        "cut_depth": str(cut),
                        "opaque_records": "true",
                        "source_truth_included": "false",
                    },
                )
                observation_entries.append(
                    artifact_entry(
                        path,
                        relative_to=public_root,
                        condition=condition,
                        cut_depth=cut,
                    )
                )

    observation_index_path = public_root / "observation_index.json"
    observation_index = {
        "schema": OBSERVATION_INDEX_SCHEMA,
        "records": [{"record_id": row["record_id"]} for row in records],
        "entries": observation_entries,
        "source_material_included": False,
    }
    validate_observation_index(observation_index)
    write_json_exclusive(observation_index_path, observation_index)

    config_path = public_root / "sanitized_config.json"
    config = {
        "schema": SANITIZED_CONFIG_SCHEMA,
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
        },
        "observation_index": artifact_entry(
            observation_index_path, relative_to=public_root
        ),
        "inverse_states": inverse_entries,
        "record_order": [row["record_id"] for row in records],
        "condition_order": list(CONDITIONS),
        "cut_order": list(CUT_DEPTHS),
        "geometry": {
            "records": 64,
            "sequence_tokens": 40,
            "scored_tokens_per_record": 39,
            "hidden_size": 2048,
            "candidate_budget": 16,
        },
        "methods": ["direct_inverse", "causal_public_surrogate_search"],
        "execution": {
            "seed": 1729,
            "stopping": "all 39 scored positions",
            "abstention": "none",
            "score_batch_size": 64,
            "causal_record_batch_size": 16,
        },
        "access_contract": "process-enforced user/mount/network/PID namespace plus minimal chroot; verified access manifest required",
        "truth_or_source_inputs": 0,
    }
    validate_sanitized_config(config)
    write_json_exclusive(config_path, config)

    truth_path = private_root / "blind_truth.jsonl"
    write_jsonl_exclusive(
        truth_path,
        (
            {"record_id": row["record_id"], "token_ids": row["token_ids"]}
            for row in records
        ),
    )
    evaluator_evidence_path = private_root / "evaluator_evidence.json"
    write_json_exclusive(
        evaluator_evidence_path,
        {
            "schema": "token-reconstruction.trr0001-r1-evaluator-evidence.v1",
            "task_id": "TRR-0001",
            "revision_id": "TRR-0001-R1",
            "command": command_record(),
            "preregistration_commit": args.preregistration_commit,
            "plan": file_record(args.plan),
            "public_commitment": file_record(args.public_commitment),
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "exit_status": 0,
            "environment": {
                "python": __import__("sys").version,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
            },
            "retained_state": {
                "target_lora": file_record(args.target_lora),
                "inverse_cut4": file_record(args.inverse_directory / "cut4.safetensors"),
                "inverse_cut8": file_record(args.inverse_directory / "cut8.safetensors"),
                "fresh_training_steps": 0,
                "fresh_adaptation_steps": 0,
            },
            "public_interface": {
                "root": str(public_root),
                "config": file_record(config_path, root=root),
                "observation_index": file_record(observation_index_path, root=root),
                "observations": observation_entries,
                "inverse_states": inverse_entries,
                "source_identity_fields": 0,
                "source_token_or_text_fields": 0,
                "opaque_records": 64,
            },
            "private_outputs": {
                "truth": file_record(truth_path, root=root),
                "frequency_counts": file_record(frequency_destination, root=root),
                "selection_mapping_rewritten": False,
            },
            "phases": timer.records,
            "peak_memory": peak_memory(),
        },
    )
    for path in sorted(public_root.rglob("*")):
        if path.is_file():
            path.chmod(0o444)
    print(
        {
            "status": "fresh_evaluator_interface_prepared",
            "records": 64,
            "scored_tokens_per_arm": 2496,
            "public_config": str(config_path),
            "source_identity_fields": 0,
            "fresh_training_steps": 0,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
