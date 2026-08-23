#!/usr/bin/env python3
"""Run the exact frozen exhaustive winner inside the owner-R1 blind namespace."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM

from token_reconstruction.a1a2_configuration_search import (
    decode_policy,
    resolved_policy_from_dict,
)
from token_reconstruction.component_crossover import propose_public_a1
from token_reconstruction.dual_benchmark import scored_mask
from token_reconstruction.experiment_runtime import (
    PhaseTimer,
    command_record,
    file_record,
    load_json,
    peak_memory,
    seed_everything,
    sha256_file,
    synchronize,
    utc_now,
    write_json_exclusive,
)


MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
METHOD_ID = "a1_a2_exhaustive_configuration_winner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--access-manifest", type=Path, required=True)
    return parser.parse_args()


def import_reference(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("trr0002_r1_blind_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import frozen public-prefix reference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def public_path(root: Path, entry: dict[str, Any]) -> Path:
    relative = str(entry["path"])
    if relative.startswith("/") or ".." in relative.split("/"):
        raise RuntimeError("public input path escaped its root")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError("public input path escaped its root") from exc
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"public input is unavailable: {relative}")
    if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
        raise RuntimeError(f"public input hash changed: {relative}")
    return path


def validate_access(value: dict[str, Any]) -> None:
    if (
        value.get("schema") != "token-reconstruction.trr0002-owner-r1-blind-isolation.v1"
        or value.get("task_id") != "TRR-0002"
        or value.get("method_id") != METHOD_ID
        or value.get("result") != "PASS_FAIL_CLOSED_ACCESS_BOUNDARY"
        or value.get("exit_status") != 0
    ):
        raise RuntimeError("blind access manifest identity changed")
    probes = value.get("denial_probes")
    if not isinstance(probes, list) or len(probes) != 9:
        raise RuntimeError("blind denial probes are incomplete")
    if any(probe.get("passed") is not True for probe in probes):
        raise RuntimeError("a blind denial probe failed")
    if value.get("permissions") != {
        "root_write_denied": True,
        "input_write_denied": True,
        "code_write_denied": True,
        "model_write_denied": True,
        "output_write_succeeded": True,
        "tmp_write_succeeded": True,
    }:
        raise RuntimeError("blind mount permission probe failed")
    if value.get("network", {}).get("passed") is not True:
        raise RuntimeError("blind network isolation failed")


def counts(values: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    unique, frequencies = torch.unique(values[mask].to(torch.long), return_counts=True)
    return {
        str(int(key.item())): int(value.item())
        for key, value in zip(unique, frequencies, strict=True)
    }


def main() -> int:
    args = parse_args()
    started_utc = utc_now()
    output_root = args.output_directory.resolve()
    if output_root.is_symlink() or not output_root.is_dir():
        raise RuntimeError("isolated output directory is invalid")
    if {path.name for path in output_root.iterdir()} != {args.access_manifest.name}:
        raise RuntimeError("isolated output was not empty before its access manifest")
    access = load_json(args.access_manifest)
    validate_access(access)
    input_root = args.input_root.resolve()
    if args.config.resolve() != input_root / "sanitized_config.json":
        raise RuntimeError("only the sanitized blind config may be supplied")
    config = load_json(args.config)
    if (
        config.get("schema")
        != "token-reconstruction.trr0002-owner-r1-blind-sanitized-config.v1"
        or config.get("task_id") != "TRR-0002"
        or config.get("method_id") != METHOD_ID
        or config.get("truth_or_source_inputs") != 0
        or config.get("geometry")
        != {
            "records": 64,
            "sequence_tokens": 40,
            "scored_tokens_per_record": 39,
            "hidden_size": 2048,
            "cut_depth": 4,
        }
        or config.get("execution", {}).get("record_batch_size") != 8
        or config.get("execution", {}).get("target_prefix_calls") != 0
    ):
        raise RuntimeError("blind sanitized config changed")
    observation_index_path = public_path(input_root, config["observation_index"])
    observation_path = public_path(input_root, config["observation"])
    lens_path = public_path(input_root, config["public_lens"])
    winner_path = public_path(input_root, config["winner"])
    winner = load_json(winner_path)
    policy = resolved_policy_from_dict(winner["policy"])
    if (
        policy.policy_id != config["policy_id"]
        or max(policy.spec.schedule) != config["execution"]["maximum_candidate_budget"]
        or policy.spec.terminal_action != config["execution"]["terminal_action"]
    ):
        raise RuntimeError("frozen policy differs from sanitized config")
    index = load_json(observation_index_path)
    if (
        index.get("schema")
        != "token-reconstruction.trr0002-owner-r1-blind-observation-index.v1"
        or index.get("source_material_included") is not False
        or index.get("observation") != config["observation"]
    ):
        raise RuntimeError("blind observation index changed")
    records = index.get("records")
    if (
        not isinstance(records, list)
        or len(records) != 64
        or [row.get("record_id") for row in records] != config["record_order"]
        or any(set(row) != {"record_id"} for row in records)
    ):
        raise RuntimeError("opaque blind record order changed")
    state = load_file(observation_path, device="cpu")
    if set(state) != {"activations"} or tuple(state["activations"].shape) != (64, 40, 2048):
        raise RuntimeError("blind observation tensor changed")
    observations = state["activations"]
    attention_mask = torch.ones((64, 40), dtype=torch.long)
    position_ids = torch.arange(40, dtype=torch.long).view(1, -1).expand(64, -1)

    seed_everything(20260826)
    timer = PhaseTimer()
    with timer.measure("load_pinned_public_teacher_and_lens"):
        expected_model_path = Path(f"/model-repo/snapshots/{MODEL_REVISION}")
        if args.model_path.resolve() != expected_model_path:
            raise RuntimeError("isolated model mount path changed")
        if not torch.cuda.is_available():
            raise RuntimeError("blind reconstruction requires CUDA")
        full = (
            AutoModelForCausalLM.from_pretrained(
                args.model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(torch.device("cuda"))
            .eval()
        )
        full.requires_grad_(False)
        if full.config.hidden_size != 2048 or full.config.vocab_size != 128256:
            raise RuntimeError("pinned model geometry changed")
        reference = import_reference(args.reference_source)
        precut = reference.PublicP0Precut(full, (0, 1, 2, 3)).to("cuda").eval()
        embeddings = reference.normalize_public_embeddings(precut.embed_tokens.weight).to("cuda")
        lens = reference.load_frozen_lens(lens_path, device=torch.device("cuda"))
        del full
    preparation_peak = peak_memory()
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    method_started_utc = utc_now()
    method_started = time.perf_counter()
    proposal = propose_public_a1(
        observations=observations,
        attention_mask=attention_mask,
        lens=lens,
        normalized_embeddings=embeddings,
    )
    selector = decode_policy(
        observations=observations,
        attention_mask=attention_mask,
        position_ids=position_ids,
        candidates=proposal.candidates,
        a1_confidence=proposal.top1_confidence,
        precut=precut,
        device=torch.device("cuda"),
        policy=policy,
        record_batch_size=8,
    )
    synchronize()
    method_seconds = time.perf_counter() - method_started
    method_ended_utc = utc_now()
    method_peak = peak_memory()
    max_k = max(policy.spec.schedule)
    prediction_path = output_root / "predictions.safetensors"
    save_file(
        {
            "predictions": selector.predictions.to(torch.int32).contiguous(),
            "candidates": proposal.candidates[:, :, :max_k].to(torch.int32).contiguous(),
            "proposal_top1_confidence": proposal.top1_confidence.float().contiguous(),
            "routes": selector.routes.to(torch.int8).contiguous(),
            "selected_k": selector.selected_k.to(torch.int16).contiguous(),
            "selected_signal": selector.selected_signal.float().contiguous(),
        },
        prediction_path,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r1-blind-prediction-freeze.v1",
            "task_id": "TRR-0002",
            "method_id": METHOD_ID,
            "policy_id": policy.policy_id,
            "truth_opened": "false",
            "maximum_candidate_budget": str(max_k),
        },
    )
    mask = scored_mask(attention_mask)
    abstained_tokens = int(selector.predictions[mask].lt(0).sum().item())
    route_path = output_root / "route.json"
    write_json_exclusive(
        route_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-blind-route.v1",
            "task_id": "TRR-0002",
            "method_id": METHOD_ID,
            "policy_id": policy.policy_id,
            "record_order": config["record_order"],
            "routes": counts(selector.routes, mask),
            "selected_k": counts(selector.selected_k, mask),
            "abstained_tokens": abstained_tokens,
            "target_prefix_calls": 0,
            "truth_or_source_inputs": 0,
            "prediction": file_record(prediction_path),
            "config": file_record(args.config),
            "access_manifest": file_record(args.access_manifest),
        },
    )
    evidence_path = output_root / "reconstructor_evidence.json"
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr0002-owner-r1-blind-reconstructor-evidence.v1",
            "task_id": "TRR-0002",
            "method_id": METHOD_ID,
            "policy_id": policy.policy_id,
            "status": "OWNER_R1_BLIND_PREDICTIONS_FROZEN",
            "command": command_record(),
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "method_started_utc": method_started_utc,
            "method_ended_utc": method_ended_utc,
            "exit_status": 0,
            "access_manifest_verified_before_inputs": True,
            "truth_or_source_inputs": 0,
            "target_prefix_calls": 0,
            "fresh_training_steps": 0,
            "fresh_adaptation_steps": 0,
            "records": 64,
            "scored_tokens": 2496,
            "proposal_seconds": proposal.elapsed_seconds,
            "selection_seconds": selector.elapsed_seconds,
            "method_compute_seconds": method_seconds,
            "logical_candidate_simulations": selector.candidate_simulations,
            "executed_candidate_simulations": selector.executed_candidate_simulations,
            "prefix_commit_tokens": selector.prefix_commit_tokens,
            "record_batch_size": selector.record_batch_size,
            "persisted_method_state": [file_record(winner_path), file_record(lens_path)],
            "memory": {"preparation_peak": preparation_peak, "method_peak_after_cuda_reset": method_peak},
            "phases": timer.records,
            "outputs": [file_record(prediction_path), file_record(route_path)],
            "source_files": [
                file_record(Path(__file__)),
                file_record(Path("/code/src/token_reconstruction/a1a2_configuration_search.py")),
                file_record(Path("/code/src/token_reconstruction/component_crossover.py")),
                file_record(args.reference_source),
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": "OWNER_R1_BLIND_PREDICTIONS_FROZEN",
                "policy_id": policy.policy_id,
                "prediction": str(prediction_path),
                "abstained_tokens": abstained_tokens,
                "truth_or_source_inputs": 0,
                "method_compute_seconds": method_seconds,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
