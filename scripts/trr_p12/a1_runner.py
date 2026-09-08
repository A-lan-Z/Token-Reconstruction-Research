#!/usr/bin/env python3
"""Run the bounded TRR-P12 native A1+A2 K256 comparator.

This is a thin P12 input/output adapter around the published P11 comparator:
``trr0004_predict_confirmation._load_public_prefix``, ``_A2Adapter``, and
``trr0004_fresh_confirmation.run_warmed_prediction`` remain the decision and
timing implementation.  The adapter consumes only four sanitized observation
files (pile/finance x stage 0/256), keeps the first 32 rows per domain, and
writes prediction, candidate-trace, cost, and receipt artifacts below an
ignored ``outputs/TRR-P12`` directory.  It never opens source text, token
IDs, labels, target weights, or evaluator truth.

The scientific run is CUDA-only because the retained public-prefix/A2 path is
CUDA-bound.  ``--check-only`` validates the input and exact P11 resource
binding without loading the prefix or starting a GPU run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


TASK_ID = "TRR-P12"
METHOD_ID = "frozen_a1_a2_k256"
SCHEMA = "token-reconstruction.trr-p12-a1-input.v1"
RUN_SCHEMA = "token-reconstruction.trr-p12-a1-run.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p12-a1-prediction.v1"
TRACE_SCHEMA = "token-reconstruction.trr-p12-a1-candidate-trace.v1"
COST_SCHEMA = "token-reconstruction.trr-p12-a1-cost.v1"
RECEIPT_SCHEMA = "token-reconstruction.trr-p12-a1-cell-receipt.v1"

DOMAINS = ("pile", "finance")
STAGES = (0, 256)
RECORDS_PER_DOMAIN = 32
STORED_SEQUENCE_TOKENS = 128
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
BOS_TOKEN_ID = 128000
DEFAULT_MINIMUM_FREE_GIB = 8.0
DEFAULT_RUNTIME_MINIMUM_FREE_GIB = 2.0
DEFAULT_MAXIMUM_RESERVED_GIB = 8.0
DEFAULT_MAXIMUM_RSS_GIB = 16.0
DEFAULT_MAX_SECONDS = 1800.0

_FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "token_ids",
        "input_ids",
        "labels",
        "target_labels",
        "target_weights",
        "source_text",
        "truth",
        "evaluator_truth",
        "record_ids",
    }
)


class A1RunnerError(RuntimeError):
    """Raised when a P12 A1 input or guarded run is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise A1RunnerError(f"{label} is not a regular file: {resolved}")
    return resolved


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label=label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A1RunnerError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise A1RunnerError(f"{label} must be a JSON object")
    return payload


def _reject_private_keys(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise A1RunnerError(f"{path} contains a non-string key")
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_MANIFEST_KEYS:
                raise A1RunnerError(f"{path}.{key} is evaluator/source-private state")
            _reject_private_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_keys(child, path=f"{path}[{index}]")


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise A1RunnerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_binding(value: Any, *, base: Path, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise A1RunnerError(f"{label} must contain a path")
    raw = Path(str(value["path"])).expanduser()
    if not raw.is_absolute():
        raw = base / raw
    path = _regular_file(raw, label=label)
    digest = _sha256_file(path)
    declared = _require_digest(value.get("sha256"), label=f"{label}.sha256")
    if digest != declared:
        raise A1RunnerError(f"{label} hash changed")
    actual_bytes = int(path.stat().st_size)
    if value.get("bytes") is not None and int(value["bytes"]) != actual_bytes:
        raise A1RunnerError(f"{label} byte count changed")
    return {"path": str(path), "bytes": actual_bytes, "sha256": digest}


def _load_observation(path: Path, *, label: str) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    path = _regular_file(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
    except Exception as exc:
        raise A1RunnerError(f"{label} is not readable safetensors") from exc
    if keys != {"activations", "attention_mask", "position_ids"}:
        raise A1RunnerError(f"{label} keys must be exactly activations, attention_mask, position_ids")
    try:
        tensors = load_file(str(path), device="cpu")
    except Exception as exc:
        raise A1RunnerError(f"{label} could not be loaded") from exc
    activation = tensors["activations"]
    mask = tensors["attention_mask"]
    positions = tensors["position_ids"]
    if activation.ndim != 3 or tuple(activation.shape[1:]) != (STORED_SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise A1RunnerError(f"{label} activation geometry must be [records,128,2048]")
    if activation.dtype != torch.bfloat16 or not bool(torch.isfinite(activation).all().item()):
        raise A1RunnerError(f"{label} activations must be finite bfloat16")
    expected = (int(activation.shape[0]), STORED_SEQUENCE_TOKENS)
    if tuple(mask.shape) != expected or tuple(positions.shape) != expected:
        raise A1RunnerError(f"{label} mask/position geometry differs from activations")
    if mask.dtype not in (torch.uint8, torch.bool, torch.int8, torch.int16, torch.int32, torch.int64):
        raise A1RunnerError(f"{label} attention mask must be boolean/integer")
    if bool(((mask != 0) & (mask != 1)).any().item()):
        raise A1RunnerError(f"{label} mask must contain only 0/1 values")
    mask = mask.to(dtype=torch.bool)
    if bool(mask[:, 0].logical_not().any().item()):
        raise A1RunnerError(f"{label} first position is masked")
    if positions.dtype != torch.int64 or not bool(torch.isfinite(positions.to(dtype=torch.float32)).all().item()):
        raise A1RunnerError(f"{label} positions must be finite int64")
    return (
        {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)},
        (activation.contiguous(), mask.contiguous(), positions.contiguous()),
    )


def _load_input_manifest(path: Path) -> dict[str, Any]:
    path = _regular_file(path, label="A1 input manifest")
    payload = _json_object(path, label="A1 input manifest")
    _reject_private_keys(payload)
    if payload.get("task_id") != TASK_ID:
        raise A1RunnerError("A1 input manifest task_id differs")
    if payload.get("schema") != SCHEMA or payload.get("status") != "SANITIZED_TARGET_OBSERVATIONS_READY":
        raise A1RunnerError("A1 input manifest schema/status is not the registered sanitized form")
    if payload.get("method_id") != METHOD_ID:
        raise A1RunnerError("A1 input manifest method identity changed")
    for flag in ("source_text_loaded", "token_ids_loaded", "target_labels_loaded", "target_weights_loaded", "truth_opened", "p03_holdout_accessed"):
        if payload.get(flag) is not False:
            raise A1RunnerError(f"A1 input manifest violates {flag}=false")
    if payload.get("domains") != list(DOMAINS) or payload.get("stages") != list(STAGES):
        raise A1RunnerError("A1 input manifest must declare pile/finance and stages 0/256 in fixed order")
    if int(payload.get("records_per_domain", -1)) != RECORDS_PER_DOMAIN:
        raise A1RunnerError("A1 input manifest records_per_domain must be 32")
    source_orders = payload.get("source_order_sha256")
    if not isinstance(source_orders, Mapping) or set(source_orders) != set(DOMAINS):
        raise A1RunnerError("A1 input manifest source-order digests are incomplete")
    source_order_digest = {domain: _require_digest(source_orders[domain], label=f"source_order_sha256/{domain}") for domain in DOMAINS}
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise A1RunnerError("A1 input manifest cells are missing")
    expected = {(domain, stage) for domain in DOMAINS for stage in STAGES}
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw in enumerate(cells):
        if not isinstance(raw, Mapping):
            raise A1RunnerError(f"A1 input cell {index} is malformed")
        domain = str(raw.get("domain"))
        stage = int(raw.get("stage", -1))
        key = (domain, stage)
        if key not in expected or key in loaded:
            raise A1RunnerError(f"A1 input cell identity is duplicate or unexpected: {domain}/{stage}")
        if raw.get("source_order_sha256") != source_order_digest[domain]:
            raise A1RunnerError(f"A1 input cell source-order binding differs: {domain}/{stage}")
        binding = _file_binding(raw.get("observation"), base=path.parent, label=f"A1 observation {domain}/{stage}")
        file_record, tensors = _load_observation(Path(binding["path"]), label=f"A1 observation {domain}/{stage}")
        if int(tensors[0].shape[0]) < RECORDS_PER_DOMAIN:
            raise A1RunnerError(f"A1 observation {domain}/{stage} has fewer than 32 rows")
        loaded[key] = {
            "domain": domain,
            "stage": stage,
            "source_order_sha256": source_order_digest[domain],
            "observation": file_record,
            "tensors": tensors,
        }
    if set(loaded) != expected:
        raise A1RunnerError(f"A1 input cells must be exactly {sorted(expected)}")
    for domain in DOMAINS:
        base = loaded[(domain, 0)]["tensors"]
        final = loaded[(domain, 256)]["tensors"]
        if not torch.equal(base[1], final[1]) or not torch.equal(base[2], final[2]):
            raise A1RunnerError(f"A1 {domain} stage masks/positions are not paired")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "schema": SCHEMA,
        "status": str(payload["status"]),
        "source_order_sha256": source_order_digest,
        "cells": loaded,
    }


def _load_binding(path: Path, *, repository_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _regular_file(path, label="P11 runtime binding")
    payload = _json_object(path, label="P11 runtime binding")
    if payload.get("schema") != "token-reconstruction.trr-p11-a1-a2-runtime-binding.v1":
        raise A1RunnerError("P11 runtime binding schema changed")
    # This validator hashes the exact preserved P11 code/resources and checks
    # the read-only truth boundary.  It does not load source rows or labels.
    try:
        from scripts.trr_p11 import a1_a2_runtime as p11_runtime

        p11_runtime.validate_runtime_binding(payload, root=repository_root)
        policy = p11_runtime.validate_native_policy(payload.get("policy", {}))
    except Exception as exc:
        raise A1RunnerError("P11 runtime binding validation failed") from exc
    resources = payload.get("resources")
    code = payload.get("code_bindings")
    if not isinstance(resources, Mapping) or not isinstance(code, Mapping):
        raise A1RunnerError("P11 runtime binding lacks code/resources")
    summary = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "schema": str(payload["schema"]),
        "method_id": str(payload.get("method_id")),
        "policy": policy,
        "code_bindings": {
            str(name): {"path": str(value.get("path")), "sha256": str(value.get("sha256")), "bytes": int(value.get("bytes"))}
            for name, value in code.items()
            if isinstance(value, Mapping)
        },
        "resources": {
            str(name): (
                {"path": str(value.get("path")), "sha256": str(value.get("sha256")), "bytes": int(value.get("bytes"))}
                if name != "model_snapshot" and isinstance(value, Mapping)
                else {
                    "path": str(value.get("path")),
                    "files": {
                        str(file_name): {"sha256": str(file_value.get("sha256")), "bytes": int(file_value.get("bytes"))}
                        for file_name, file_value in value.get("files", {}).items()
                        if isinstance(file_value, Mapping)
                    },
                }
            )
            for name, value in resources.items()
            if isinstance(value, Mapping)
        },
        "truth_boundary": {
            "source_text_loaded": False,
            "token_ids_written": False,
            "target_labels_loaded": False,
            "truth_opened": False,
            "p03_holdout_accessed": False,
        },
    }
    return payload, summary


def _guard_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        minimum_free_gib=float(args.minimum_free_gib),
        maximum_reserved_gib=float(args.maximum_reserved_gib),
        maximum_rss_gib=float(args.maximum_rss_gib),
        max_seconds=float(args.max_seconds),
    )


def _create_only(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise A1RunnerError(f"{label} is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = _create_only(path, label=label)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _resolve_script_modules(repository_root: Path) -> tuple[Any, Any, Any, Any]:
    scripts_dir = repository_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import trr0003_footing_compare as footing
        import trr0004_predict_confirmation as legacy
        from scripts.trr_p11 import a1_a2_runtime as p11_runtime
    except Exception as exc:
        raise A1RunnerError("preserved P11/P12 runtime modules are unavailable") from exc
    return footing, legacy, legacy.fc, p11_runtime


def _cell_name(domain: str, stage: int) -> str:
    return f"{domain}_stage{int(stage):04d}"


def _load_qualification_resume(path: Path, *, output: Path, expected_cell: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _regular_file(path, label="A1 qualification receipt")
    if path.parent.resolve() != output.resolve():
        raise A1RunnerError("qualification receipt and resume output root must match")
    payload = _json_object(path, label="A1 qualification receipt")
    if payload.get("schema") != RUN_SCHEMA or payload.get("status") != "A1_A2_QUALIFICATION_COMPLETE_NO_TRUTH":
        raise A1RunnerError("resume input is not a completed A1 qualification receipt")
    if payload.get("qualification_only") is not True or payload.get("qualification_cell") != expected_cell:
        raise A1RunnerError("resume qualification cell or mode changed")
    for flag in ("truth_opened", "source_text_loaded", "token_ids_loaded", "target_labels_loaded", "target_weights_loaded", "p03_holdout_accessed"):
        if payload.get(flag) is not False:
            raise A1RunnerError(f"qualification receipt violates {flag}=false")
    cells = payload.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != {expected_cell}:
        raise A1RunnerError("qualification receipt must contain exactly one retained cell")
    cell = cells[expected_cell]
    if not isinstance(cell, Mapping):
        raise A1RunnerError("qualification cell payload is malformed")
    for key in ("prediction", "trace", "cost", "receipt"):
        binding = cell.get(key)
        if not isinstance(binding, Mapping):
            raise A1RunnerError(f"qualification cell lacks {key} binding")
        _file_binding(binding, base=output, label=f"qualification/{expected_cell}/{key}")
    return payload, {str(expected_cell): dict(cell)}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(args.repository_root).expanduser().resolve()
    input_payload = _load_input_manifest(Path(args.input_manifest))
    binding, binding_summary = _load_binding(Path(args.binding), repository_root=repository_root)
    if args.check_only:
        return {
            "schema": RUN_SCHEMA,
            "task_id": TASK_ID,
            "status": "CHECK_ONLY_PASS_NO_MODEL_OR_GPU",
            "created_utc": _utc_now(),
            "input_manifest": {key: value for key, value in input_payload.items() if key != "cells"},
            "input_cells": sorted(f"{domain}/{stage}" for domain, stage in input_payload["cells"]),
            "binding": binding_summary,
            "device": str(args.device),
            "model_loaded": False,
            "target_weights_loaded": False,
            "truth_opened": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
        }

    if torch.device(str(args.device)).type != "cuda":
        raise A1RunnerError("native P12 A1 run requires CUDA; use --check-only for CPU validation")
    if args.qualification_only and args.resume_qualification is not None:
        raise A1RunnerError("qualification-only and resume cannot be combined")
    qualification_cell = str(args.qualification_cell)
    expected_cells = {_cell_name(domain, stage) for domain in DOMAINS for stage in STAGES}
    if qualification_cell not in expected_cells:
        raise A1RunnerError(f"unknown qualification cell: {qualification_cell}")
    footing, legacy, fc, p11_runtime = _resolve_script_modules(repository_root)
    if int(getattr(legacy, "DEFAULT_A2_K", -1)) != 256 or int(getattr(legacy, "DEFAULT_A2_PROPOSAL_K", -1)) != 512 or int(getattr(legacy, "DEFAULT_A1_CHUNK", -1)) != 256:
        raise A1RunnerError("preserved native A1/A2 K256 constants changed")
    device = torch.device(str(args.device))
    output = Path(args.output_root).expanduser().resolve()
    resume_payload: dict[str, Any] | None = None
    retained_cells: dict[str, Any] = {}
    if args.resume_qualification is not None:
        if not output.is_dir() or output.is_symlink():
            raise A1RunnerError("resume output root must be the existing qualification directory")
        resume_payload, retained_cells = _load_qualification_resume(
            Path(args.resume_qualification), output=output, expected_cell=qualification_cell
        )
    else:
        if output.exists() or output.is_symlink():
            raise A1RunnerError(f"A1 output root is create-only: {output}")
        output.mkdir(parents=True, exist_ok=False)
    all_cell_ids = [_cell_name(domain, stage) for domain in DOMAINS for stage in STAGES]
    if args.qualification_only:
        cells_to_run = (qualification_cell,)
    elif args.resume_qualification is not None:
        cells_to_run = tuple(cell_id for cell_id in all_cell_ids if cell_id != qualification_cell)
    else:
        cells_to_run = tuple(all_cell_ids)
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    guard_args = _guard_args(args)
    runtime_guard_args = _guard_args(args)
    runtime_guard_args.minimum_free_gib = float(args.runtime_minimum_free_gib)
    preflight: list[dict[str, Any]] = []
    watchdog_process = None
    watchdog_path = None
    watchdog_command = None
    run_cells: dict[str, Any] = dict(retained_cells)
    try:
        preflight.append(legacy._resource_preflight(guard_args, device, stage="before_a1_load", started=started_clock))
        resources = binding["resources"]
        snapshot = Path(str(resources["model_snapshot"]["path"]))
        reference = Path(str(resources["public_reference"]["path"]))
        lens = Path(str(resources["retained_a1_lens"]["path"]))
        embedding = Path(str(resources["public_embedding_table"]["path"]))
        watchdog_process, watchdog_path, watchdog_command = p11_runtime._start_external_watchdog(
            root=repository_root,
            output=output,
            device=device,
            maximum_rss_gib=float(args.maximum_rss_gib),
            minimum_free_gpu_gib=float(args.runtime_minimum_free_gib),
            maximum_seconds=float(args.max_seconds),
            phase=("a1_a2_qualification" if args.qualification_only else ("a1_a2_resume" if args.resume_qualification is not None else "a1_a2")),
        )
        load_started = time.perf_counter()
        precut, lens_module, embeddings, public_evidence = legacy._load_public_prefix(
            snapshot=snapshot,
            reference_path=reference,
            lens_path=lens,
            embedding_path=embedding,
            device=device,
        )
        load_seconds = time.perf_counter() - load_started
        policy = p11_runtime.validate_native_policy(footing._fixed_k256_policy())
        adapter = legacy._A2Adapter(
            precut=precut,
            lens=lens_module,
            embeddings=embeddings,
            device=device,
            policy=policy,
        )
        preflight.append(legacy._resource_preflight(runtime_guard_args, device, stage="after_a1_load", started=started_clock))
        for domain in DOMAINS:
            for stage in STAGES:
                cell_id = _cell_name(domain, stage)
                if cell_id not in cells_to_run:
                    continue
                cell = input_payload["cells"][(domain, stage)]
                activations, mask, positions = cell["tensors"]
                activations = activations[:RECORDS_PER_DOMAIN].contiguous()
                mask = mask[:RECORDS_PER_DOMAIN].contiguous()
                positions = positions[:RECORDS_PER_DOMAIN].contiguous()
                cell_preflight_start = len(preflight)
                preflight.append(legacy._resource_preflight(runtime_guard_args, device, stage=f"before_{cell_id}", started=started_clock))
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                adapter.begin_cell()
                timing_started = time.perf_counter()
                predictions, timing = fc.run_warmed_prediction(
                    observations=activations,
                    attention_mask=mask,
                    position_ids=positions,
                    predictor=adapter,
                    device=device,
                    warmup_runs=1,
                    measured_runs=3,
                )
                timing_elapsed = time.perf_counter() - timing_started
                if any(not bool(record.get("repeated_prediction_exact")) for record in timing.get("records", [])):
                    raise A1RunnerError(f"A1 repeated prediction mismatch: {cell_id}")
                if tuple(predictions.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS):
                    raise A1RunnerError(f"A1 prediction geometry changed: {cell_id}")
                if bool(predictions[:, 0].ne(BOS_TOKEN_ID).any().item()) or bool(predictions.lt(0).any().item()) or bool(predictions.ge(VOCABULARY_SIZE).any().item()):
                    raise A1RunnerError(f"A1 prediction range/BOS check failed: {cell_id}")
                candidates, candidate_scores = adapter.candidate_tensors(records=RECORDS_PER_DOMAIN, sequence_tokens=STORED_SEQUENCE_TOKENS)
                if tuple(candidates.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS, 512) or tuple(candidate_scores.shape) != tuple(candidates.shape):
                    raise A1RunnerError(f"A1 candidate trace geometry changed: {cell_id}")
                predictions = predictions.detach().cpu().to(dtype=torch.long).contiguous()
                candidates = candidates.detach().cpu().to(dtype=torch.int64).contiguous()
                candidate_scores = candidate_scores.detach().cpu().to(dtype=torch.float32).contiguous()
                peak = legacy._peak_memory(device)
                adapter_evidence = adapter.evidence()
                preflight.append(legacy._resource_preflight(runtime_guard_args, device, stage=f"after_{cell_id}", started=started_clock))
                prediction_path = output / "predictions" / f"{cell_id}.safetensors"
                trace_path = output / "traces" / f"{cell_id}.safetensors"
                save_file(
                    {"predictions": predictions},
                    str(_create_only(prediction_path, label=f"A1 prediction {cell_id}")),
                    metadata={
                        "schema": PREDICTION_SCHEMA,
                        "task_id": TASK_ID,
                        "method_id": METHOD_ID,
                        "domain": domain,
                        "stage": str(stage),
                        "records": str(RECORDS_PER_DOMAIN),
                        "stored_sequence_tokens": str(STORED_SEQUENCE_TOKENS),
                        "candidate_budget": "256",
                        "proposal_budget": "512",
                        "source_text_loaded": "false",
                        "token_ids_loaded": "false",
                        "target_labels_loaded": "false",
                        "truth_opened": "false",
                    },
                )
                save_file(
                    {"candidates": candidates, "candidate_scores": candidate_scores},
                    str(_create_only(trace_path, label=f"A1 candidate trace {cell_id}")),
                    metadata={
                        "schema": TRACE_SCHEMA,
                        "task_id": TASK_ID,
                        "method_id": METHOD_ID,
                        "domain": domain,
                        "stage": str(stage),
                        "records": str(RECORDS_PER_DOMAIN),
                        "stored_sequence_tokens": str(STORED_SEQUENCE_TOKENS),
                        "candidate_budget": "256",
                        "proposal_budget": "512",
                        "truth_opened": "false",
                    },
                )
                prediction_binding = {"path": str(prediction_path), "bytes": int(prediction_path.stat().st_size), "sha256": _sha256_file(prediction_path), "key": "predictions"}
                trace_binding = {"path": str(trace_path), "bytes": int(trace_path.stat().st_size), "sha256": _sha256_file(trace_path), "keys": ["candidates", "candidate_scores"]}
                cost_payload = {
                    "schema": COST_SCHEMA,
                    "task_id": TASK_ID,
                    "method_id": METHOD_ID,
                    "cell_id": cell_id,
                    "domain": domain,
                    "stage": int(stage),
                    "timing": timing,
                    "timing_elapsed_seconds_wrapper": timing_elapsed,
                    "adapter": adapter_evidence,
                    "peak_memory": peak,
                    "resource_preflight": preflight[cell_preflight_start:],
                    "truth_opened": False,
                    "source_text_loaded": False,
                    "token_ids_loaded": False,
                    "target_labels_loaded": False,
                    "target_weights_loaded": False,
                    "p03_holdout_accessed": False,
                }
                cost_binding = _write_json(output / "costs" / f"{cell_id}.json", cost_payload, label=f"A1 cost receipt {cell_id}")
                receipt_payload = {
                    "schema": RECEIPT_SCHEMA,
                    "task_id": TASK_ID,
                    "status": "A1_A2_CELL_COMPLETE_NO_TRUTH",
                    "method_id": METHOD_ID,
                    "cell_id": cell_id,
                    "domain": domain,
                    "stage": int(stage),
                    "records": RECORDS_PER_DOMAIN,
                    "source_order_sha256": cell["source_order_sha256"],
                    "prediction": prediction_binding,
                    "trace": trace_binding,
                    "cost": cost_binding,
                    "candidate_policy": policy,
                    "truth_opened": False,
                    "source_text_loaded": False,
                    "token_ids_loaded": False,
                    "target_labels_loaded": False,
                    "target_weights_loaded": False,
                    "p03_holdout_accessed": False,
                    "input_observation": cell["observation"],
                }
                receipt_binding = _write_json(output / "receipts" / f"{cell_id}.json", receipt_payload, label=f"A1 prediction receipt {cell_id}")
                run_cells[cell_id] = {
                    "domain": domain,
                    "stage": int(stage),
                    "records": RECORDS_PER_DOMAIN,
                    "source_order_sha256": cell["source_order_sha256"],
                    "prediction": prediction_binding,
                    "trace": trace_binding,
                    "cost": cost_binding,
                    "receipt": receipt_binding,
                    "timing": timing,
                    "peak_memory": peak,
                    "adapter": adapter_evidence,
                }
        final_preflight = legacy._resource_preflight(runtime_guard_args, device, stage="after_a1_all_cells", started=started_clock)
        preflight.append(final_preflight)
        status = "A1_A2_QUALIFICATION_COMPLETE_NO_TRUTH" if args.qualification_only else "A1_A2_RUN_COMPLETE_NO_TRUTH"
    except Exception:
        raise
    finally:
        watchdog_receipt = None
        if watchdog_process is not None:
            watchdog_receipt = p11_runtime._stop_external_watchdog(
                watchdog_process,
                watchdog_path,
                root=repository_root,
                command=watchdog_command,
            )
    aggregate = {
        "schema": RUN_SCHEMA,
        "task_id": TASK_ID,
        "status": status,
        "created_utc": started_utc,
        "ended_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started_clock,
        "method_id": METHOD_ID,
        "scope": {
            "domains": list(DOMAINS),
            "stages": list(STAGES),
            "records_per_domain": RECORDS_PER_DOMAIN,
            "executed_cells": list(cells_to_run),
            "source_subset": "first32 rows from each sanitized domain/stage input; no source IDs consumed",
            "timing": "one warmup plus three measured exact-repeat calls per record; first measured prediction retained",
        },
        "qualification_only": bool(args.qualification_only),
        "qualification_cell": qualification_cell,
        "resumed_from_qualification": (
            {"path": str(Path(args.resume_qualification).expanduser().resolve()), "bytes": int(Path(args.resume_qualification).expanduser().resolve().stat().st_size), "sha256": _sha256_file(Path(args.resume_qualification).expanduser().resolve())}
            if args.resume_qualification is not None else None
        ),
        "input_manifest": {key: value for key, value in input_payload.items() if key != "cells"},
        "input_cells": {cell_id: {key: value for key, value in cell.items() if key != "tensors"} for cell_id, cell in ((f"{d}_{s}", input_payload["cells"][(d, s)]) for d in DOMAINS for s in STAGES)},
        "binding": binding_summary,
        "runner_code": {"path": str(Path(__file__).resolve()), "bytes": int(Path(__file__).stat().st_size), "sha256": _sha256_file(Path(__file__).resolve())},
        "public_prefix_load_seconds": load_seconds,
        "public_prefix_evidence": public_evidence,
        "policy": policy,
        "cells": run_cells,
        "resource_preflight": preflight,
        "watchdog": watchdog_receipt,
        "guards": {
            "initial_minimum_free_gpu_gib": float(args.minimum_free_gib),
            "runtime_minimum_free_gpu_gib": float(args.runtime_minimum_free_gib),
            "maximum_reserved_gpu_gib": float(args.maximum_reserved_gib),
            "maximum_host_rss_gib": float(args.maximum_rss_gib),
            "maximum_wall_seconds": float(args.max_seconds),
        },
        "device": str(device),
        "truth_opened": False,
        "source_text_loaded": False,
        "token_ids_loaded": False,
        "target_labels_loaded": False,
        "target_weights_loaded": False,
        "p03_holdout_accessed": False,
    }
    return aggregate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path, help="published P11 runtime-binding.json")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/TRR-P12/a1-a2-r1"))
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_GIB, help="preload free-GPU guard")
    parser.add_argument("--runtime-minimum-free-gib", type=float, default=DEFAULT_RUNTIME_MINIMUM_FREE_GIB, help="post-load free-GPU/watchdog guard")
    parser.add_argument("--maximum-reserved-gib", type=float, default=DEFAULT_MAXIMUM_RESERVED_GIB)
    parser.add_argument("--maximum-rss-gib", type=float, default=DEFAULT_MAXIMUM_RSS_GIB)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--qualification-only", action="store_true", help="run and retain one representative cell before the remaining cells")
    parser.add_argument("--qualification-cell", default="finance_stage0000", help="cell retained by qualification-only and skipped on resume")
    parser.add_argument("--resume-qualification", type=Path, help="resume the existing qualification output root without rerunning its retained cell")
    parser.add_argument("--check-only", action="store_true", help="validate manifest and P11 binding without model/GPU loading")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
        if args.check_only:
            print(json.dumps(result, sort_keys=True))
        else:
            output = Path(args.output_root).expanduser().resolve()
            receipt_name = "run.resume.receipt.json" if args.resume_qualification is not None else "run.receipt.json"
            _write_json(output / receipt_name, result, label="A1 aggregate receipt")
            print(json.dumps({"status": result["status"], "output_root": str(output), "cells": sorted(result["cells"])}, sort_keys=True))
        return 0
    except (A1RunnerError, RuntimeError, ValueError, OSError) as exc:
        print(f"a1_runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
