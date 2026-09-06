#!/usr/bin/env python3
"""Run or preflight the bounded P04 native A1+A2 K=256 anchor.

The runner reuses the immutable PR7 proposal/decode implementation.  It
consumes only a truth-free evaluator observation artifact and its metadata;
the source token IDs are never loaded.  ``--preflight-only`` validates the
12-record, 384-position anchor and expected cost without loading any model,
target update, or observation tensor.  A real run is one target condition at
a time and writes P04 prediction rows plus cost evidence in a create-only
directory.

This anchor is an exact algorithmic reuse on a new P04 input subset.  Because
the P04 panel has three styles and a different geometry from PR7, results are
reported as a panel/input port and never as a canonical dual-benchmark run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

from scripts.trr_p04.prepare_evaluator_observations import (
    CONDITIONS,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MAXIMUM_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    _load_object,
    _row_order_sha256,
    _validate_selection,
    _validate_target_plan,
    _write_create_only,
)


TASK_ID = "TRR-P04"
SCHEMA = "token-reconstruction.trr-p04-native-anchor.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p04-predictions.v1"
METHOD_ID = "native_a1_a2"
IMPLEMENTATION_ID = "frozen_a1_a2_k256"
PROPOSAL_K = 512
PROPOSAL_CHUNK = 256
CANDIDATE_K = 256
RECORD_BATCH_SIZE = 1
WARMUP_PASSES = 1
MEASURED_PASSES = 3
ANCHOR_RECORDS = 12
ANCHOR_LENGTH = 32
EXPECTED_POST_BOS_POSITIONS = ANCHOR_RECORDS * ANCHOR_LENGTH
PR7_ROOT = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004"
)
DEFAULT_REFERENCE = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/"
    "configuration-search/fresh-blind-code/reference/strict_bos/round001_teacher.py"
)
DEFAULT_LENS = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/"
    "blind/reconstructor_input/public_a1_lens.pt"
)
PR7_LEGACY = PR7_ROOT / "scripts/trr0003_footing_compare.py"
PR7_POLICY = PR7_ROOT / "src/token_reconstruction/a1a2_configuration_search.py"
PR7_TARGET_UPDATE = PR7_ROOT / "src/token_reconstruction/target_update.py"
MIN_FREE_GPU_BYTES = 8 * 2**30
MAX_RESERVED_GPU_BYTES = 6 * 2**30
MAX_HOST_RSS_BYTES = 16 * 2**30
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


class NativeAnchorError(RuntimeError):
    """Raised when the native anchor cannot be bound or completed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _safe_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _git_head() -> str | None:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _descriptor(path: Path, *, role: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise NativeAnchorError(f"{role} is unavailable: {path}")
    return {"role": role, "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _validate_anchor_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    anchors = [dict(row) for row in rows if bool(row.get("anchor"))]
    if len(anchors) != ANCHOR_RECORDS:
        raise NativeAnchorError(f"native anchor requires {ANCHOR_RECORDS} records")
    for style in ("pile_plain", "finance_chat", "alpaca_instruction"):
        cell = [row for row in rows if row.get("style") == style and int(row.get("length_stratum", -1)) == ANCHOR_LENGTH]
        selected = [row for row in anchors if row.get("style") == style]
        if len(cell) != 6 or len(selected) != 4 or selected != cell[:4]:
            raise NativeAnchorError(f"native anchor selection changed for {style}")
    if any(int(row.get("length_stratum", -1)) != ANCHOR_LENGTH for row in anchors):
        raise NativeAnchorError("native anchor contains a non-32-token record")
    return anchors


def build_preflight(
    *,
    selection_path: Path,
    target_plan_path: Path,
    output_root: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    selection = _load_object(selection_path, label="P04 selection")
    rows, panel = _validate_selection(selection, selection_path=selection_path)
    anchors = _validate_anchor_rows(rows)
    target_plan = _load_object(target_plan_path, label="evaluator target plan")
    target = _validate_target_plan(target_plan, selection=selection)
    value = {
        "schema": f"{SCHEMA}-preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS_NO_MODEL_NO_TARGET_NO_TRUTH",
        "created_utc": _utc_now(),
        "selection": {**panel, "path": str(selection_path.resolve()), "sha256": _sha256_file(selection_path)},
        "anchor": {
            "method_id": METHOD_ID,
            "implementation_identity": IMPLEMENTATION_ID,
            "port_status": "exact PR7 algorithm on P04 panel/input port; not canonical dual-benchmark claim",
            "record_count": len(anchors),
            "record_order_sha256": _row_order_sha256(anchors),
            "post_bos_length_per_record": ANCHOR_LENGTH,
            "scored_positions_per_target": EXPECTED_POST_BOS_POSITIONS,
            "denominator_separate": True,
        },
        "algorithm": {
            "proposal_k": PROPOSAL_K,
            "proposal_chunk": PROPOSAL_CHUNK,
            "candidate_k": CANDIDATE_K,
            "record_batch_size": RECORD_BATCH_SIZE,
            "policy": "fixed schedule [256], direct cosine, no fast path, commit last winner",
            "a2_fallback": False,
            "tie_rule": "published proposal order and first argmax",
            "expected_candidate_simulations": EXPECTED_POST_BOS_POSITIONS * CANDIDATE_K,
            "expected_proposal_positions": EXPECTED_POST_BOS_POSITIONS,
        },
        "target_conditions": list(CONDITIONS),
        "target_plan": {"path": str(target_plan_path.resolve()), "sha256": _sha256_file(target_plan_path), **target},
        "reusable_sources": [
            _descriptor(PR7_LEGACY, role="immutable PR7 legacy proposal/decode"),
            _descriptor(PR7_POLICY, role="immutable PR7 policy definitions"),
        ],
        "access": {
            "source_rows_read": False,
            "observation_tensor_read": False,
            "model_loaded": False,
            "target_update_loaded": False,
            "evaluation_truth_opened": False,
        },
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "git_commit": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise NativeAnchorError(f"preflight output must be new and empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    _write_create_only(output_root / "native_anchor_preflight.json", value)
    return value


def _load_module(path: Path, name: str) -> Any:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise NativeAnchorError(f"{name} source is unavailable: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise NativeAnchorError(f"unable to load {name} source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_guard(device: torch.device, *, stage: str, started: float) -> dict[str, Any]:
    if device.type != "cuda":
        raise NativeAnchorError("native A1+A2 anchor requires CUDA")
    try:
        free, total = torch.cuda.mem_get_info(device)
        reserved = int(torch.cuda.memory_reserved(device))
    except Exception as exc:
        raise NativeAnchorError(f"CUDA resource data unavailable at {stage}") from exc
    rss = _max_rss_bytes()
    if free < MIN_FREE_GPU_BYTES or reserved > MAX_RESERVED_GPU_BYTES or rss > MAX_HOST_RSS_BYTES:
        raise NativeAnchorError(f"resource guard failed at {stage}: free={free}, reserved={reserved}, rss={rss}")
    return {
        "stage": stage,
        "status": "PASS",
        "free_bytes": int(free),
        "total_bytes": int(total),
        "reserved_bytes": reserved,
        "rss_bytes": rss,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _load_observation(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise NativeAnchorError(f"observation artifact is unavailable: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            forbidden = keys.intersection({"token_ids", "input_ids", "labels", "source_text", "truth", "oracle"})
            if forbidden:
                raise NativeAnchorError(f"observation artifact contains forbidden fields: {sorted(forbidden)}")
            required = {"activations", "attention_mask", "position_ids"}
            if not required.issubset(keys):
                raise NativeAnchorError(f"observation artifact lacks required keys: {sorted(required - keys)}")
            activations = handle.get_tensor("activations").contiguous()
            mask = handle.get_tensor("attention_mask").contiguous()
            positions = handle.get_tensor("position_ids").contiguous()
            metadata = dict(handle.metadata() or {})
    except NativeAnchorError:
        raise
    except Exception as exc:
        raise NativeAnchorError(f"cannot load truth-free observations: {path}") from exc
    if tuple(activations.shape) != (72, MAXIMUM_TOKENS, HIDDEN_SIZE):
        raise NativeAnchorError(f"observation geometry changed: {tuple(activations.shape)}")
    if mask.shape != activations.shape[:2] or positions.shape != activations.shape[:2]:
        raise NativeAnchorError("observation mask/position geometry changed")
    if mask.dtype not in (torch.uint8, torch.bool) or positions.dtype not in (torch.int64, torch.int32):
        raise NativeAnchorError("observation mask/position dtype changed")
    if not torch.isfinite(activations.float()).all().item():
        raise NativeAnchorError("observation activation is non-finite")
    mask_bool = mask.to(torch.bool)
    # The capture contract is fixed right padding: BOS is active, active
    # positions form one prefix, and position IDs are zero-based within that
    # prefix with zeroes only in the padded suffix.  Validate this before any
    # anchor selection or native proposal/decode call so a malformed artifact
    # cannot silently change the denominator or expose later source tokens.
    if not bool(mask_bool[:, 0].all().item()):
        raise NativeAnchorError("observation rows must keep BOS active")
    if not torch.equal(mask_bool, mask_bool.cumprod(dim=1).to(torch.bool)):
        raise NativeAnchorError("observation mask must be contiguous right padding")
    positions_long = positions.to(torch.long)
    for row_index in range(positions_long.shape[0]):
        active = int(mask_bool[row_index].sum().item())
        expected_active = torch.arange(active, dtype=torch.long)
        if not torch.equal(positions_long[row_index, :active].cpu(), expected_active):
            raise NativeAnchorError(f"observation position IDs are not contiguous at row {row_index}")
        if active < positions_long.shape[1] and not bool(
            torch.all(positions_long[row_index, active:] == 0).item()
        ):
            raise NativeAnchorError(f"observation padded position IDs are not zero at row {row_index}")
    return activations, mask_bool, positions_long, metadata


def _anchor_indices(index_path: Path, rows: Sequence[Mapping[str, Any]]) -> list[int]:
    index = _load_object(index_path, label="observation index")
    index_rows = index.get("records")
    if not isinstance(index_rows, list) or len(index_rows) != len(rows):
        raise NativeAnchorError("observation index does not cover the frozen panel")
    selected: list[int] = []
    for position, (expected, actual) in enumerate(zip(rows, index_rows)):
        if not isinstance(actual, Mapping) or actual.get("record_id") != expected.get("record_id"):
            raise NativeAnchorError(f"observation index order changed at row {position}")
        if bool(actual.get("anchor")):
            selected.append(position)
    if selected != [index for index, row in enumerate(rows) if bool(row.get("anchor"))]:
        raise NativeAnchorError("observation index anchor flags changed")
    if len(selected) != ANCHOR_RECORDS:
        raise NativeAnchorError("observation index has wrong anchor count")
    return selected


def _load_reference_resources(
    *,
    model_snapshot: Path,
    reference_path: Path,
    lens_path: Path,
    condition: str,
    device: torch.device,
) -> tuple[Any, Any, torch.Tensor, dict[str, Any]]:
    """Load the same public reference resources for both paired conditions.

    The shifted observation condition is represented by its activation artifact.
    The anchor's causal prefix scorer remains the untouched public base model;
    it must never load or hash the private evaluator target update.
    """
    from transformers import AutoModelForCausalLM

    if condition not in CONDITIONS:
        raise NativeAnchorError(f"unknown target condition: {condition}")
    reference = _load_module(reference_path, "trr_p04_native_reference")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_snapshot.expanduser().resolve()),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        if int(model.config.hidden_size) != HIDDEN_SIZE or int(model.config.vocab_size) != 128256:
            raise NativeAnchorError("native anchor model geometry changed")
        model.requires_grad_(False)
        precut = reference.PublicP0Precut(model, (0, 1, 2, 3)).to(device).eval()
        embeddings = reference.normalize_public_embeddings(precut.embed_tokens.weight).to(device)
        lens = reference.load_frozen_lens(lens_path.expanduser().resolve(), device=device)
    except NativeAnchorError:
        raise
    except Exception as exc:
        raise NativeAnchorError("native reference resource loading failed") from exc
    target_descriptor: dict[str, Any] = {
        "condition": condition,
        "public_reference_loaded": True,
        "evaluator_target_update_loaded": False,
        "target_update_weights_available_to_reconstructor": False,
        "public_reference_identity": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_snapshot": str(model_snapshot.expanduser().resolve()),
            "prefix_layers": [0, 1, 2, 3],
            "reference_path": str(reference_path.expanduser().resolve()),
            "lens_path": str(lens_path.expanduser().resolve()),
        },
    }
    return precut, lens, embeddings, target_descriptor


def run_anchor(
    *,
    selection_path: Path,
    target_plan_path: Path,
    observation_index_path: Path,
    observation_path: Path,
    model_snapshot: Path,
    reference_path: Path,
    lens_path: Path,
    condition: str,
    output_root: Path,
    device: torch.device,
    argv: Sequence[str],
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise NativeAnchorError(f"unknown target condition: {condition}")
    started_perf = time.perf_counter()
    selection = _load_object(selection_path, label="P04 selection")
    rows, panel = _validate_selection(selection, selection_path=selection_path)
    anchors = _validate_anchor_rows(rows)
    target_plan = _load_object(target_plan_path, label="evaluator target plan")
    target = _validate_target_plan(target_plan, selection=selection)
    if output_root.exists():
        if output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir()):
            raise NativeAnchorError(f"anchor output must be new and empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    guards: list[dict[str, Any]] = []
    guards.append(_cuda_guard(device, stage="before_observation_load", started=started_perf))
    activations, masks, positions, observation_metadata = _load_observation(observation_path)
    if observation_metadata.get("condition") != condition:
        raise NativeAnchorError(
            "observation condition metadata does not match the requested native anchor condition"
        )
    if observation_metadata.get("record_order_sha256") != panel["record_order_sha256"]:
        raise NativeAnchorError("observation record order is not bound to the frozen panel")
    if observation_metadata.get("selection_sha256") != panel["sha256"]:
        raise NativeAnchorError("observation selection is not bound to the frozen panel")
    selected_indices = _anchor_indices(observation_index_path, rows)
    if _row_order_sha256([rows[index] for index in selected_indices]) != _row_order_sha256(anchors):
        raise NativeAnchorError("anchor row order differs from frozen panel")
    anchor_h = activations[selected_indices]
    anchor_mask = masks[selected_indices]
    anchor_positions = positions[selected_indices]
    if not anchor_mask[:, : ANCHOR_LENGTH + 1].all().item():
        raise NativeAnchorError("anchor observations do not cover BOS plus 32 scored positions")
    guards.append(_cuda_guard(device, stage="after_observation_load", started=started_perf))
    load_started = time.perf_counter()
    precut, lens, embeddings, target_descriptor = _load_reference_resources(
        model_snapshot=model_snapshot,
        reference_path=reference_path,
        lens_path=lens_path,
        condition=condition,
        device=device,
    )
    load_seconds = time.perf_counter() - load_started
    guards.append(_cuda_guard(device, stage="after_reference_resource_load", started=started_perf))
    legacy = _load_module(PR7_LEGACY, "trr0003_footing_compare")
    policy = legacy._fixed_k256_policy()
    output_lines: list[str] = []
    timing_rows: list[dict[str, Any]] = []
    logical_proposal_seconds_total = 0.0
    executed_proposal_seconds_total = 0.0
    logical_candidate_simulations_total = 0
    logical_executed_simulations_total = 0
    executed_candidate_simulations_total = 0
    executed_simulations_total = 0
    logical_prefix_commit_tokens_total = 0
    prefix_commit_tokens_total = 0
    logical_prefix_calls_total = 0
    prefix_calls_total = 0
    logical_prediction_seconds_total = 0.0
    measured_prediction_seconds_total = 0.0
    executed_prediction_seconds_total = 0.0
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    guards.append(_cuda_guard(device, stage="before_timed_anchor_passes", started=started_perf))

    for local_index, (record, row_h, row_mask, row_positions) in enumerate(
        zip(anchors, anchor_h, anchor_mask, anchor_positions)
    ):
        if tuple(row_h.shape) != (MAXIMUM_TOKENS, HIDDEN_SIZE):
            raise NativeAnchorError("anchor activation row geometry changed")
        observations = row_h.view(1, MAXIMUM_TOKENS, HIDDEN_SIZE)
        attention_mask = row_mask.view(1, MAXIMUM_TOKENS).to(torch.long)
        position_ids = row_positions.view(1, MAXIMUM_TOKENS).to(torch.long)

        def run_pass() -> dict[str, Any]:
            _synchronize(device)
            prefix_before = int(getattr(precut, "checked_cache_transitions", 0))
            pass_started = time.perf_counter()
            proposal = legacy.propose_public_a1(
                observations=observations,
                attention_mask=attention_mask,
                lens=lens,
                normalized_embeddings=embeddings,
                max_k=PROPOSAL_K,
                chunk=PROPOSAL_CHUNK,
            )
            decoded = legacy.decode_policy(
                observations=observations,
                attention_mask=attention_mask,
                position_ids=position_ids,
                candidates=proposal.candidates[:, :, :CANDIDATE_K].contiguous(),
                a1_confidence=proposal.top1_confidence,
                precut=precut,
                device=device,
                policy=policy,
                record_batch_size=RECORD_BATCH_SIZE,
            )
            _synchronize(device)
            elapsed = time.perf_counter() - pass_started
            if tuple(decoded.predictions.shape) != (1, MAXIMUM_TOKENS):
                raise NativeAnchorError("native anchor prediction geometry changed")
            predictions = decoded.predictions[0, 1 : ANCHOR_LENGTH + 1].detach().cpu().to(torch.long).tolist()
            if len(predictions) != ANCHOR_LENGTH or any(int(value) < 0 or int(value) >= 128256 for value in predictions):
                raise NativeAnchorError("native anchor emitted invalid prediction IDs")
            prefix_after = int(getattr(precut, "checked_cache_transitions", 0))
            if prefix_after < prefix_before:
                raise NativeAnchorError("native public prefix transition counter moved backwards")
            result = {
                "predictions": [int(value) for value in predictions],
                "elapsed_seconds": float(elapsed),
                "proposal_seconds": float(proposal.elapsed_seconds),
                "candidate_simulations": int(decoded.candidate_simulations),
                "executed_candidate_simulations": int(decoded.executed_candidate_simulations),
                "prefix_commit_tokens": int(decoded.prefix_commit_tokens),
                "public_prefix_calls": prefix_after - prefix_before,
            }
            del proposal, decoded
            return result

        passes = [run_pass() for _ in range(WARMUP_PASSES + MEASURED_PASSES)]
        baseline_predictions = passes[0]["predictions"]
        if any(current["predictions"] != baseline_predictions for current in passes[1:]):
            raise NativeAnchorError("native anchor predictions changed across warmup or measured repeats")
        warmup = passes[:WARMUP_PASSES]
        measured = passes[WARMUP_PASSES:]
        logical = measured[0]
        warmup_elapsed = sum(float(item["elapsed_seconds"]) for item in warmup)
        measured_elapsed = sum(float(item["elapsed_seconds"]) for item in measured)
        executed_elapsed = sum(float(item["elapsed_seconds"]) for item in passes)
        logical_proposal_seconds_total += float(logical["proposal_seconds"])
        executed_proposal_seconds_total += sum(float(item["proposal_seconds"]) for item in passes)
        logical_candidate_simulations_total += int(logical["candidate_simulations"])
        logical_executed_simulations_total += int(logical["executed_candidate_simulations"])
        executed_candidate_simulations_total += sum(int(item["candidate_simulations"]) for item in passes)
        executed_simulations_total += sum(int(item["executed_candidate_simulations"]) for item in passes)
        logical_prefix_commit_tokens_total += int(logical["prefix_commit_tokens"])
        prefix_commit_tokens_total += sum(int(item["prefix_commit_tokens"]) for item in passes)
        logical_prefix_calls_total += int(logical["public_prefix_calls"])
        prefix_calls_total += sum(int(item["public_prefix_calls"]) for item in passes)
        logical_prediction_seconds_total += float(logical["elapsed_seconds"])
        measured_prediction_seconds_total += measured_elapsed
        executed_prediction_seconds_total += executed_elapsed

        def timing_summary(item: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "elapsed_seconds": float(item["elapsed_seconds"]),
                "proposal_seconds": float(item["proposal_seconds"]),
                "candidate_simulations": int(item["candidate_simulations"]),
                "executed_candidate_simulations": int(item["executed_candidate_simulations"]),
                "prefix_commit_tokens": int(item["prefix_commit_tokens"]),
                "public_prefix_calls": int(item["public_prefix_calls"]),
            }

        output_lines.append(
            json.dumps(
                {
                    "schema": PREDICTION_SCHEMA,
                    "method_id": METHOD_ID,
                    "seed": None,
                    "condition": condition,
                    "record_id": str(record["record_id"]),
                    "predicted_token_ids": baseline_predictions,
                    "anchor": True,
                },
                sort_keys=True,
            )
        )
        timing_rows.append(
            {
                "record_id": str(record["record_id"]),
                "post_bos_positions": ANCHOR_LENGTH,
                "warmup_passes": WARMUP_PASSES,
                "measured_repeat_passes": MEASURED_PASSES,
                "prediction_exact_repeat": True,
                "warmup": [timing_summary(item) for item in warmup],
                "measured_repeats": [timing_summary(item) for item in measured],
                "logical_one_pass": timing_summary(logical),
                "warmup_elapsed_seconds": warmup_elapsed,
                "measured_elapsed_seconds": measured_elapsed,
                "executed_elapsed_seconds_including_warmup_repeats": executed_elapsed,
            }
        )
        guards.append(_cuda_guard(device, stage=f"after_anchor_row_{local_index:02d}", started=started_perf))
    _synchronize(device)
    prediction_path = output_root / "predictions.jsonl"
    _write_create_only_text(prediction_path, "\n".join(output_lines) + "\n")
    diagnostics = {
        "schema": f"{SCHEMA}-diagnostics.v1",
        "task_id": TASK_ID,
        "status": "PASS_NATIVE_ANCHOR_NO_TRUTH",
        "method_id": METHOD_ID,
        "implementation_identity": IMPLEMENTATION_ID,
        "condition": condition,
        "port_status": "exact PR7 algorithm on P04 panel/input port; not canonical dual-benchmark claim",
        "record_count": ANCHOR_RECORDS,
        "scored_positions": EXPECTED_POST_BOS_POSITIONS,
        "proposal_k": PROPOSAL_K,
        "candidate_k": CANDIDATE_K,
        "candidate_simulations": logical_candidate_simulations_total,
        "executed_candidate_simulations": logical_executed_simulations_total,
        "warmup_passes": WARMUP_PASSES,
        "measured_repeat_passes": MEASURED_PASSES,
        "prediction_exact_repeat": True,
        "executed_candidate_simulations_including_warmup_repeats": executed_candidate_simulations_total,
        "executed_candidate_simulations_including_warmup_repeats_actual": executed_simulations_total,
        "logical_proposal_seconds_sum": logical_proposal_seconds_total,
        "executed_proposal_seconds_including_warmup_repeats": executed_proposal_seconds_total,
        "logical_prefix_commit_tokens": logical_prefix_commit_tokens_total,
        "executed_prefix_commit_tokens_including_warmup_repeats": prefix_commit_tokens_total,
        "logical_public_prefix_calls": logical_prefix_calls_total,
        "executed_public_prefix_calls_including_warmup_repeats": prefix_calls_total,
        "record_batch_size": RECORD_BATCH_SIZE,
        "a2_fallback": False,
        "tie_rule": "published proposal order and first argmax",
        "timing_rows": timing_rows,
    }
    diagnostics_path = output_root / "native_anchor_diagnostics.json"
    _write_create_only(diagnostics_path, diagnostics)
    receipt = {
        "schema": f"{SCHEMA}-receipt.v1",
        "task_id": TASK_ID,
        "status": "PASS_NATIVE_ANCHOR_NO_TRUTH",
        "created_utc": _utc_now(),
        "selection": {**panel, "path": str(selection_path.resolve()), "sha256": _sha256_file(selection_path)},
        "target_plan": {"path": str(target_plan_path.resolve()), "sha256": _sha256_file(target_plan_path), **target},
        "observation": {
            "path": str(observation_path.resolve()),
            "sha256": _sha256_file(observation_path),
            "metadata": observation_metadata,
            "index_path": str(observation_index_path.resolve()),
            "index_sha256": _sha256_file(observation_index_path),
        },
        "target": target_descriptor,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(model_snapshot.resolve())},
        "resources": {
            "reference": _descriptor(reference_path, role="immutable PR7 reference implementation"),
            "lens": _descriptor(lens_path, role="retained public A1 lens"),
            "legacy_proposal_decode": _descriptor(PR7_LEGACY, role="immutable PR7 legacy proposal/decode"),
            "policy": _descriptor(PR7_POLICY, role="immutable PR7 policy definitions"),
        },
        "algorithm": {
            "method_id": METHOD_ID,
            "implementation_identity": IMPLEMENTATION_ID,
            "proposal_k": PROPOSAL_K,
            "proposal_chunk": PROPOSAL_CHUNK,
            "candidate_k": CANDIDATE_K,
            "record_batch_size": RECORD_BATCH_SIZE,
            "expected_candidate_simulations": EXPECTED_POST_BOS_POSITIONS * CANDIDATE_K,
            "actual_candidate_simulations": logical_candidate_simulations_total,
            "actual_executed_candidate_simulations": logical_executed_simulations_total,
            "warmup_passes": WARMUP_PASSES,
            "measured_repeat_passes": MEASURED_PASSES,
            "prediction_exact_repeat": True,
            "executed_candidate_simulations_including_warmup_repeats": executed_candidate_simulations_total,
            "executed_candidate_simulations_including_warmup_repeats_actual": executed_simulations_total,
            "a2_fallback": False,
            "tie_rule": "published proposal order and first argmax",
        },
        "prediction": {
            "path": str(prediction_path),
            "bytes": int(prediction_path.stat().st_size),
            "sha256": _sha256_file(prediction_path),
            "rows": ANCHOR_RECORDS,
            "post_bos_positions": EXPECTED_POST_BOS_POSITIONS,
        },
        "diagnostics": {
            "path": str(diagnostics_path),
            "bytes": int(diagnostics_path.stat().st_size),
            "sha256": _sha256_file(diagnostics_path),
        },
        "timing": {
            "resource_load_seconds": load_seconds,
            "logical_one_pass_prediction_seconds": logical_prediction_seconds_total,
            "measured_repeat_prediction_seconds": measured_prediction_seconds_total,
            "executed_prediction_seconds_including_warmup_repeats": executed_prediction_seconds_total,
            "logical_one_pass_proposal_seconds": logical_proposal_seconds_total,
            "executed_proposal_seconds_including_warmup_repeats": executed_proposal_seconds_total,
            "logical_one_pass_public_prefix_calls": logical_prefix_calls_total,
            "executed_public_prefix_calls_including_warmup_repeats": prefix_calls_total,
            "logical_one_pass_prefix_commit_tokens": logical_prefix_commit_tokens_total,
            "executed_prefix_commit_tokens_including_warmup_repeats": prefix_commit_tokens_total,
            "warmup_passes_per_record": WARMUP_PASSES,
            "measured_repeat_passes_per_record": MEASURED_PASSES,
        },
        "memory": {
            "cuda_peak_allocated_bytes_after_timed_pass_reset": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes_after_timed_pass_reset": int(torch.cuda.max_memory_reserved(device)),
            "host_max_rss_bytes": _max_rss_bytes(),
            "peak_scope": "timed anchor passes after CUDA peak-counter reset; external watchdog covers whole process group",
        },
        "guards": guards,
        "access": {
            "source_rows_read": False,
            "source_tokens_read": False,
            "evaluation_truth_opened": False,
            "public_reference_loaded": True,
            "evaluator_target_update_loaded": False,
            "target_update_loaded": False,
            "student_states_loaded": False,
        },
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "git_commit": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "elapsed_seconds": round(time.perf_counter() - started_perf, 6),
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    _write_create_only(output_root / "native_anchor_receipt.json", receipt)
    del precut, lens, embeddings
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


def _write_create_only_text(path: Path, value: str) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise NativeAnchorError(f"refusing to overwrite create-only output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--observation-index", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--model-snapshot", type=Path, required=False, default=Path(
        "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
        "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    ))
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--condition", choices=CONDITIONS, default="public_base")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    effective_argv = list(sys.argv if argv is None else [sys.argv[0], *argv])
    try:
        if args.preflight_only:
            value = build_preflight(
                selection_path=args.selection.expanduser().resolve(),
                target_plan_path=args.target_plan.expanduser().resolve(),
                output_root=args.output_root.expanduser().resolve(),
                argv=effective_argv,
            )
        else:
            if args.observation_index is None or args.observations is None:
                raise NativeAnchorError("a real anchor run requires --observation-index and --observations")
            value = run_anchor(
                selection_path=args.selection.expanduser().resolve(),
                target_plan_path=args.target_plan.expanduser().resolve(),
                observation_index_path=args.observation_index.expanduser().resolve(),
                observation_path=args.observations.expanduser().resolve(),
                model_snapshot=args.model_snapshot.expanduser().resolve(),
                reference_path=args.reference.expanduser().resolve(),
                lens_path=args.lens.expanduser().resolve(),
                condition=args.condition,
                output_root=args.output_root.expanduser().resolve(),
                device=torch.device(args.device),
                argv=effective_argv,
            )
        print(json.dumps({"status": value["status"], "output_root": str(args.output_root.resolve())}, sort_keys=True))
        return 0
    except (NativeAnchorError, RuntimeError) as exc:
        print(f"P04 native anchor failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
