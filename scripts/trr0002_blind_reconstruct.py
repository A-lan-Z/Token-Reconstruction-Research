#!/usr/bin/env python3
"""Run the frozen calibrated selector inside the TRR-0002 blind chroot."""

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

from token_reconstruction.calibrated_selector import select_calibrated_adaptive
from token_reconstruction.component_crossover import propose_public_a1
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
CALIBRATION_SHA256 = "ad1801ec348a61cbcd50bfbc4a991c8deaa503b79f454c7f1d779567042ebf47"
LENS_SHA256 = "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742"
METHOD_ID = "a1_scale_calibrated_adaptive_causal_k32_to64"


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
    spec = importlib.util.spec_from_file_location("trr0002_blind_reference", path)
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
        value.get("schema") != "token-reconstruction.trr0002-blind-isolation.v1"
        or value.get("task_id") != "TRR-0002"
        or value.get("method_id") != METHOD_ID
        or value.get("result") != "PASS_FAIL_CLOSED_ACCESS_BOUNDARY"
        or value.get("exit_status") != 0
    ):
        raise RuntimeError("blind access manifest identity changed")
    probes = value.get("denial_probes")
    if not isinstance(probes, list) or len(probes) != 8:
        raise RuntimeError("blind denial probes are incomplete")
    if any(probe.get("passed") is not True for probe in probes):
        raise RuntimeError("a blind denial probe failed")
    permissions = value.get("permissions")
    if permissions != {
        "root_write_denied": True,
        "input_write_denied": True,
        "code_write_denied": True,
        "model_write_denied": True,
        "output_write_succeeded": True,
        "tmp_write_succeeded": True,
    }:
        raise RuntimeError("blind mount permission probe failed")
    network = value.get("network")
    if not isinstance(network, dict) or network.get("passed") is not True:
        raise RuntimeError("blind network isolation failed")


def validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != "token-reconstruction.trr0002-blind-sanitized-config.v1"
        or config.get("task_id") != "TRR-0002"
        or config.get("method_id") != METHOD_ID
        or config.get("truth_or_source_inputs") != 0
    ):
        raise RuntimeError("blind sanitized config identity changed")
    if config.get("geometry") != {
        "records": 64,
        "sequence_tokens": 40,
        "scored_tokens_per_record": 39,
        "hidden_size": 2048,
        "cut_depth": 4,
    }:
        raise RuntimeError("blind sanitized geometry changed")
    execution = config.get("execution")
    if execution != {
        "seed": 2801,
        "base_budget": 32,
        "maximum_budget": 64,
        "record_batch_size": 8,
        "threshold": 1.2544946670532227,
        "abstention": "none",
        "stopping": "all 39 scored positions",
        "target_prefix_calls": 0,
    }:
        raise RuntimeError("blind execution rule changed")


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
    validate_config(config)
    observation_index_path = public_path(input_root, config["observation_index"])
    observation_path = public_path(input_root, config["observation"])
    lens_path = public_path(input_root, config["public_lens"])
    calibration_path = public_path(input_root, config["calibration"])
    if sha256_file(lens_path) != LENS_SHA256 or sha256_file(calibration_path) != CALIBRATION_SHA256:
        raise RuntimeError("frozen public method state changed")
    calibration = load_json(calibration_path)
    if (
        calibration.get("schema") != "token-reconstruction.trr0002-frozen-calibration.v1"
        or calibration.get("threshold") != config["execution"]["threshold"]
        or calibration.get("method_id") != METHOD_ID
    ):
        raise RuntimeError("frozen calibration fields changed")
    index = load_json(observation_index_path)
    if (
        index.get("schema") != "token-reconstruction.trr0002-blind-observation-index.v1"
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

    seed_everything(2801)
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
        embeddings = reference.normalize_public_embeddings(
            precut.embed_tokens.weight
        ).to("cuda")
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
    selector = select_calibrated_adaptive(
        observations=observations,
        attention_mask=attention_mask,
        position_ids=position_ids,
        candidates=proposal.candidates[:, :, :64].contiguous(),
        precut=precut,
        device=torch.device("cuda"),
        threshold=float(calibration["threshold"]),
        record_batch_size=8,
    )
    synchronize()
    method_seconds = time.perf_counter() - method_started
    method_ended_utc = utc_now()
    method_peak = peak_memory()

    prediction_path = output_root / "predictions.safetensors"
    save_file(
        {
            "predictions": selector.predictions.to(torch.int32).contiguous(),
            "candidates_k64": proposal.candidates[:, :, :64].to(torch.int32).contiguous(),
            "proposal_top1_confidence": proposal.top1_confidence.float().contiguous(),
            "base_selection_scores": selector.base_scores.float().contiguous(),
            "extra_selection_scores": selector.extra_scores.float().contiguous(),
            "normalized_gap": selector.normalized_gap.float().contiguous(),
            "routes": selector.routes.to(torch.int8).contiguous(),
        },
        prediction_path,
        metadata={
            "schema": "token-reconstruction.trr0002-blind-prediction-freeze.v1",
            "task_id": "TRR-0002",
            "method_id": METHOD_ID,
            "truth_opened": "false",
            "threshold": format(float(calibration["threshold"]), ".17g"),
        },
    )
    route_path = output_root / "route.json"
    write_json_exclusive(
        route_path,
        {
            "schema": "token-reconstruction.trr0002-blind-route.v1",
            "task_id": "TRR-0002",
            "method_id": METHOD_ID,
            "record_order": config["record_order"],
            "threshold": calibration["threshold"],
            "base_budget": 32,
            "maximum_budget": 64,
            "expanded_positions": int(selector.routes[:, 1:].eq(3).sum().item()),
            "abstained_tokens": 0,
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
            "schema": "token-reconstruction.trr0002-blind-reconstructor-evidence.v1",
            "task_id": "TRR-0002",
            "method_id": METHOD_ID,
            "status": "BLIND_PREDICTIONS_FROZEN",
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
            "base_candidate_simulations": selector.base_candidate_simulations,
            "extra_candidate_simulations": selector.extra_candidate_simulations,
            "logical_candidate_simulations": (
                selector.base_candidate_simulations
                + selector.extra_candidate_simulations
            ),
            "executed_candidate_simulations": selector.executed_candidate_simulations,
            "prefix_commit_tokens": selector.prefix_commit_tokens,
            "persisted_method_state": [
                file_record(calibration_path),
                file_record(lens_path),
            ],
            "memory": {
                "preparation_peak": preparation_peak,
                "method_peak_after_cuda_reset": method_peak,
            },
            "phases": timer.records,
            "outputs": [file_record(prediction_path), file_record(route_path)],
            "source_files": [
                file_record(Path(__file__)),
                file_record(
                    Path("/code/src/token_reconstruction/calibrated_selector.py")
                ),
                file_record(args.reference_source),
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": "BLIND_PREDICTIONS_FROZEN",
                "prediction": str(prediction_path),
                "expanded_positions": int(selector.routes[:, 1:].eq(3).sum().item()),
                "abstained_tokens": 0,
                "truth_or_source_inputs": 0,
                "method_compute_seconds": method_seconds,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
