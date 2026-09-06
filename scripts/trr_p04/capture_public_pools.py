#!/usr/bin/env python3
"""Capture P04 public correction/validation activations with the pinned prefix.

The selector has already frozen the source rows and rendering hashes.  This
adapter materializes those public rows, verifies the rendering and truncated
sequence hashes against that receipt, and reuses the immutable PR7
``ContiguousPublicPrefix`` capture path at the existing 192-token geometry.
It writes public labels alongside activations because these pools are public
training data.  It never opens evaluator truth, target-update weights, or the
fresh evaluation panel rows.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from scripts.trr_p04 import prepare_panel as selector
from token_reconstruction.public_activation import (
    CUT_DEPTH,
    HIDDEN_SIZE,
    VOCAB_SIZE,
    capture_public_prefix,
    make_artifact_metadata,
    pad_public_token_sequences,
    save_public_artifact,
    tensor_sha256,
    validate_activation_tensor,
    validate_padded_token_batch,
)


TASK_ID = "TRR-P04"
CAPTURE_SCHEMA = "token-reconstruction.trr-p04-public-pool-capture.v1"
RECORD_SCHEMA = "token-reconstruction.trr-p04-public-pool-activation-records.v1"
MAXIMUM_TOKENS = 192
PAD_TOKEN_ID = 128001
MODEL_ID = selector.MODEL_ID
MODEL_REVISION = selector.MODEL_REVISION
DEFAULT_MODEL_SNAPSHOT = Path(selector.TOKENIZER_SNAPSHOT)
PR7_CAPTURE_HELPER = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/scripts/"
    "trr0004_prepare_public_activations.py"
)
KNOWN_MODEL_WEIGHT_BYTES = 2_471_645_608
KNOWN_MODEL_WEIGHT_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"
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


class CaptureError(RuntimeError):
    """Raised when public activation preparation fails closed."""


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
    import os

    return {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"{label} must be an object")
    return value


def _file_descriptor(path: Path, *, role: str, allow_symlink: bool = False) -> dict[str, Any]:
    original = path.expanduser()
    if original.is_symlink() and not allow_symlink:
        raise CaptureError(f"{role} cannot be a symlink: {original}")
    resolved = original.resolve()
    if not resolved.is_file():
        raise CaptureError(f"{role} is unavailable: {original}")
    return {
        "role": role,
        "path": str(original.absolute()),
        "resolved_path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
        "symlink": original.is_symlink(),
    }


def _load_pr7_helper() -> Any:
    helper_path = PR7_CAPTURE_HELPER.expanduser().resolve()
    if not helper_path.is_file() or helper_path.is_symlink():
        raise CaptureError(f"immutable PR7 capture helper is unavailable: {helper_path}")
    spec = importlib.util.spec_from_file_location("trr_p04_pr7_capture_helper", helper_path)
    if spec is None or spec.loader is None:
        raise CaptureError("unable to import immutable PR7 capture helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_paths(selection: Mapping[str, Any]) -> dict[str, tuple[Path, ...]]:
    sources = selection.get("sources")
    if not isinstance(sources, Mapping):
        raise CaptureError("selection source descriptors are absent")
    result: dict[str, tuple[Path, ...]] = {}
    for style in selector.STYLES:
        descriptor = sources.get(style)
        files = descriptor.get("arrow_files") if isinstance(descriptor, Mapping) else None
        if not isinstance(files, list) or not files:
            raise CaptureError(f"selection has no Arrow descriptor for {style}")
        paths: list[Path] = []
        for row in files:
            if not isinstance(row, Mapping):
                raise CaptureError(f"selection Arrow descriptor for {style} is malformed")
            raw = row.get("resolved_path") or row.get("path")
            if not isinstance(raw, str) or not raw:
                raise CaptureError(f"selection Arrow path for {style} is absent")
            path = Path(raw).expanduser().resolve()
            if not path.is_file() or path.is_symlink():
                raise CaptureError(f"selection Arrow source is unavailable: {path}")
            paths.append(path)
        result[style] = tuple(paths)
    return result


def _load_public_datasets(selection: Mapping[str, Any]) -> dict[str, Any]:
    paths = _source_paths(selection)
    result: dict[str, Any] = {}
    try:
        from datasets import concatenate_datasets

        for style in selector.STYLES:
            parts = [Dataset.from_file(str(path)) for path in paths[style]]
            result[style] = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    except Exception as exc:
        raise CaptureError("unable to open the pinned public Arrow sources") from exc
    return result


def _finance_tokens(row: Mapping[str, Any], tokenizer: Any) -> list[int]:
    system, user, assistant = selector._finance_fields(row)
    if not user or not assistant:
        raise CaptureError("Finance public row lacks user/assistant text")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            date_string="06 Aug 2026",
        )
    except Exception as exc:
        raise CaptureError("Finance public chat rendering failed") from exc
    values = selector._tokenizer_ids(encoded)
    if not values or values[0] != selector.BOS_TOKEN_ID:
        raise CaptureError("Finance public rendering lost the declared BOS token")
    return values


def _tokens_for_row(style: str, row: Mapping[str, Any], tokenizer: Any) -> list[int]:
    if style == "pile_plain":
        text = selector._text_value(row, "text")
        values = [selector.BOS_TOKEN_ID, *selector._tokenizer_ids(tokenizer(text, add_special_tokens=False))]
    elif style == "finance_chat":
        values = _finance_tokens(row, tokenizer)
    elif style == "alpaca_instruction":
        from token_reconstruction.alpaca_split import historical_rendered_text

        rendered = historical_rendered_text(row, tokenizer)
        values = selector._tokenizer_ids(tokenizer(rendered, add_special_tokens=False))
        if not values or values[0] != selector.BOS_TOKEN_ID:
            raise CaptureError("Alpaca public rendering lost the declared BOS token")
    else:
        raise CaptureError(f"unsupported public style: {style}")
    if len(values) < 2 or values[0] != selector.BOS_TOKEN_ID:
        raise CaptureError(f"public {style} row has invalid BOS/length")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= VOCAB_SIZE for value in values):
        raise CaptureError(f"public {style} row contains an invalid token ID")
    return [int(value) for value in values]


def _materialize_pool(
    selection: Mapping[str, Any],
    *,
    pool_name: str,
    datasets: Mapping[str, Any],
    tokenizer: Any,
    selection_sha256: str,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, int]]:
    pools = selection.get("pools")
    pool = pools.get(pool_name) if isinstance(pools, Mapping) else None
    source_rows = pool.get("records") if isinstance(pool, Mapping) else None
    if not isinstance(source_rows, list) or not source_rows:
        raise CaptureError(f"selection {pool_name} records are unavailable")
    sequences: list[list[int]] = []
    manifest_rows: list[dict[str, Any]] = []
    truncated = 0
    for ordinal, declared in enumerate(source_rows):
        if not isinstance(declared, Mapping):
            raise CaptureError(f"selection {pool_name} row {ordinal} is malformed")
        style = declared.get("style")
        row_index = declared.get("row_index")
        if style not in selector.STYLES or isinstance(row_index, bool) or not isinstance(row_index, int):
            raise CaptureError(f"selection {pool_name} row {ordinal} has invalid style/index")
        dataset = datasets[style]
        if row_index < 0 or row_index >= len(dataset):
            raise CaptureError(f"selection {pool_name} row {ordinal} is outside its Arrow source")
        source_row = dataset[row_index]
        candidate = selector._candidate_from_row(style, row_index, source_row, tokenizer)
        if candidate is None:
            raise CaptureError(f"selection {pool_name} row {ordinal} became invalid")
        values = _tokens_for_row(style, source_row, tokenizer)
        checks = {
            "record_id": candidate.record_id,
            "public_record_sha256": candidate.public_record_sha256,
            "truncated_sequence_sha256": candidate.truncated_sequence_sha256,
            "full_token_count": candidate.full_token_count,
            "post_bos_token_count": candidate.post_bos_token_count,
        }
        for key, actual in checks.items():
            if str(declared.get(key)) != str(actual):
                raise CaptureError(f"selection {pool_name} source binding changed for {candidate.record_id}: {key}")
        captured = values[:MAXIMUM_TOKENS]
        if len(captured) != min(len(values), MAXIMUM_TOKENS):
            raise CaptureError("public sequence truncation is inconsistent")
        if len(values) > MAXIMUM_TOKENS:
            truncated += 1
        row = dict(declared)
        row["selection_sha256"] = selection_sha256
        row["original_full_token_count"] = int(candidate.full_token_count)
        row["original_post_bos_token_count"] = int(candidate.post_bos_token_count)
        row["full_token_count"] = len(captured)
        row["post_bos_token_count"] = len(captured) - 1
        row["padded_length"] = MAXIMUM_TOKENS
        row["capture_truncated_to_maximum_tokens"] = len(values) > MAXIMUM_TOKENS
        manifest_rows.append(row)
        sequences.append(captured)
    return sequences, manifest_rows, {"records": len(manifest_rows), "truncated_records": truncated}


def _write_record_manifest(path: Path, *, pool: str, rows: Sequence[Mapping[str, Any]], selection_path: Path, selection_sha256: str) -> dict[str, Any]:
    value = {
        "schema": RECORD_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_ACTIVATION_RECORDS_READY_NO_EVALUATION_TRUTH",
        "pool": pool,
        "record_count": len(rows),
        "records": [dict(row) for row in rows],
        "current_token_alignment": "activations[record,position] predicts token_ids[record,position]",
        "maximum_tokens": MAXIMUM_TOKENS,
        "source_text_included": False,
        "token_ids_included": False,
        "evaluation_truth_included": False,
        "selection": {"path": str(selection_path.resolve()), "sha256": selection_sha256},
    }
    if path.exists() or path.is_symlink():
        raise CaptureError(f"record manifest is create-only: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_public_artifact(
    path: Path,
    *,
    batch: Any,
    activations: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    pool: str,
    selection_sha256: str,
    source_digest: str,
    source_metadata_digest: str,
) -> dict[str, Any]:
    validate_padded_token_batch(batch, maximum_tokens=MAXIMUM_TOKENS, pad_token_id=PAD_TOKEN_ID)
    validate_activation_tensor(activations, batch, hidden_size=HIDDEN_SIZE)
    metadata = make_artifact_metadata(
        split=f"p04_{pool}",
        source_plan_sha256=selection_sha256,
        source_arrow_sha256=source_digest,
        source_info_sha256=source_metadata_digest,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        cut_depth=CUT_DEPTH,
        token_batch=batch,
        activations=activations,
        records=records,
    )
    metadata.update(
        {
            "p04_selection_sha256": selection_sha256,
            "p04_pool": pool,
            "p04_maximum_tokens": str(MAXIMUM_TOKENS),
            "source_text_materialized_transiently": "true",
            "target_weights_accessed": "false",
            "evaluator_private_truth_accessed": "false",
        }
    )
    save_public_artifact(path, activations=activations, token_batch=batch, metadata=metadata)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "activation_tensor_sha256": tensor_sha256(activations),
        "token_ids_tensor_sha256": tensor_sha256(batch.token_ids),
        "shape": list(activations.shape),
    }


@contextmanager
def _phase(phases: list[dict[str, Any]], name: str) -> Iterator[None]:
    started = time.perf_counter()
    started_utc = _utc_now()
    status = "PASS"
    error: str | None = None
    try:
        yield
    except Exception as exc:
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        row: dict[str, Any] = {
            "phase": name,
            "status": status,
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
        if error is not None:
            row["error"] = error
        phases.append(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-records", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--min-free-gpu-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--max-reserved-gpu-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--max-host-rss-bytes", type=int, default=16 * 1024**3)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_utc = _utc_now()
    started = time.perf_counter()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise CaptureError(f"capture output root is create-only and already exists: {output_root}")
    if args.batch_records <= 0 or args.batch_records > 8 or args.threads <= 0:
        raise CaptureError("P04 capture batch/thread limits are invalid")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise CaptureError("CUDA capture was requested but is unavailable")
    output_root.mkdir(parents=True)
    phases: list[dict[str, Any]] = []
    selection_path = args.selection.expanduser().resolve()
    selection = _load_json(selection_path, label="P04 frozen public selection")
    if selection.get("schema") != selector.SCHEMA or selection.get("task_id") != TASK_ID:
        raise CaptureError("P04 selection identity changed")
    selection_sha256 = _sha256_file(selection_path)
    pool_counts: dict[str, dict[str, int]] = {}
    artifact_descriptors: dict[str, dict[str, Any]] = {}
    manifest_descriptors: dict[str, dict[str, Any]] = {}
    device = torch.device(args.device)
    helper = None
    prefix = None
    try:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.interop_threads)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True, warn_only=False)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with _phase(phases, "public_resource_preflight"):
            helper = _load_pr7_helper()
            guard = helper._resource_preflight(
                device,
                min_free_gpu_bytes=args.min_free_gpu_bytes,
                max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                max_host_rss_bytes=args.max_host_rss_bytes,
            )
        with _phase(phases, "load_public_sources_and_materialize_labels"):
            tokenizer_path = args.tokenizer.expanduser().resolve()
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True)
            if getattr(tokenizer, "bos_token_id", None) != selector.BOS_TOKEN_ID:
                raise CaptureError("pinned tokenizer BOS ID changed")
            datasets = _load_public_datasets(selection)
            materialized: dict[str, tuple[list[list[int]], list[dict[str, Any]], dict[str, int]]] = {}
            for pool_name in ("correction", "validation"):
                materialized[pool_name] = _materialize_pool(
                    selection,
                    pool_name=pool_name,
                    datasets=datasets,
                    tokenizer=tokenizer,
                    selection_sha256=selection_sha256,
                )
                pool_counts[pool_name] = materialized[pool_name][2]
        source_digest = selector.json_sha256(selection.get("sources", {}))
        source_metadata_digest = selector.json_sha256(
            {"model": selection.get("model", {}), "tokenizer": selection.get("tokenizer", {}), "sources": selection.get("sources", {})}
        )
        with _phase(phases, "load_pinned_public_prefix"):
            prefix, model_snapshot, model_config = helper._load_public_prefix(
                args.model_snapshot.expanduser().resolve(),
                device=device,
                cut_depth=CUT_DEPTH,
            )
            helper._enforce_resource_ceiling(
                device,
                max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                max_host_rss_bytes=args.max_host_rss_bytes,
            )
        all_sequences = [row for pool_name in ("correction", "validation") for row in materialized[pool_name][0]]
        qualification_sequences = [row for row in all_sequences if len(row) < MAXIMUM_TOKENS][: args.batch_records]
        if len(qualification_sequences) < 2:
            raise CaptureError("public capture qualification needs at least two padded rows")
        qualification_batch = pad_public_token_sequences(qualification_sequences, maximum_tokens=MAXIMUM_TOKENS, pad_token_id=PAD_TOKEN_ID)
        with _phase(phases, "qualify_fixed_batch8_x_192_padding"):
            qualification = helper._qualify_public_prefix_padding(
                prefix,
                qualification_batch,
                device=device,
                batch_size=min(args.batch_records, len(qualification_sequences)),
            )
        captured: dict[str, tuple[Any, torch.Tensor]] = {}
        for pool_name in ("correction", "validation"):
            sequences, rows, _ = materialized[pool_name]
            batch = pad_public_token_sequences(sequences, maximum_tokens=MAXIMUM_TOKENS, pad_token_id=PAD_TOKEN_ID)
            with _phase(phases, f"capture_{pool_name}_public_prefix"):
                activations = capture_public_prefix(
                    prefix,
                    batch,
                    device=device,
                    batch_size=args.batch_records,
                    resource_check=lambda: helper._enforce_resource_ceiling(
                        device,
                        max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                        max_host_rss_bytes=args.max_host_rss_bytes,
                    ),
                )
            validate_activation_tensor(activations, batch, hidden_size=HIDDEN_SIZE)
            captured[pool_name] = (batch, activations)
            manifest_path = output_root / f"{pool_name}_records.json"
            manifest_descriptors[pool_name] = _write_record_manifest(
                manifest_path,
                pool=pool_name,
                rows=rows,
                selection_path=selection_path,
                selection_sha256=selection_sha256,
            )
        with _phase(phases, "serialize_public_pool_artifacts"):
            for pool_name, (batch, activations) in captured.items():
                artifact_descriptors[pool_name] = _write_public_artifact(
                    output_root / f"{pool_name}_cut4.safetensors",
                    batch=batch,
                    activations=activations,
                    records=materialized[pool_name][1],
                    pool=pool_name,
                    selection_sha256=selection_sha256,
                    source_digest=source_digest,
                    source_metadata_digest=source_metadata_digest,
                )
                helper._enforce_resource_ceiling(
                    device,
                    max_reserved_gpu_bytes=args.max_reserved_gpu_bytes,
                    max_host_rss_bytes=args.max_host_rss_bytes,
                )
        final_peak = helper._peak_memory(device)
        source_helper = _file_descriptor(PR7_CAPTURE_HELPER, role="immutable PR7 capture helper", allow_symlink=False)
        source_adapter = _file_descriptor(Path(__file__).resolve(), role="P04 capture adapter", allow_symlink=False)
        evidence = {
            "schema": CAPTURE_SCHEMA,
            "task_id": TASK_ID,
            "status": "PUBLIC_CORRECTION_VALIDATION_CAPTURE_COMPLETE_NO_EVALUATION_TRUTH",
            "selection": {"path": str(selection_path), "bytes": selection_path.stat().st_size, "sha256": selection_sha256},
            "pools": pool_counts,
            "geometry": {"maximum_tokens": MAXIMUM_TOKENS, "hidden_size": HIDDEN_SIZE, "vocab_size": VOCAB_SIZE, "cut_depth": CUT_DEPTH, "batch_records": args.batch_records},
            "access_contract": {
                "public_labels_read": True,
                "source_text_transient_only": True,
                "model_loaded": True,
                "target_weights_accessed": False,
                "evaluation_truth_accessed": False,
                "fresh_evaluation_records_accessed": False,
                "network_used": False,
            },
            "source": {
                "model": model_snapshot,
                "model_config": model_config,
                "tokenizer_snapshot": str(args.tokenizer.expanduser().resolve()),
                "pr7_capture_helper": source_helper,
                "p04_capture_adapter": source_adapter,
            },
            "resource_guard": guard,
            "qualification": qualification,
            "peak_memory": final_peak,
            "outputs": {"artifacts": artifact_descriptors, "record_manifests": manifest_descriptors},
            "execution": {
                "argv": list(sys.argv),
                "python": sys.version,
                "platform": platform.platform(),
                "safe_environment": _safe_environment(),
                "device": str(device),
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "max_rss_bytes": _max_rss_bytes(),
                "source_code_commit": _git_head(),
            },
            "phases": phases,
        }
        evidence_path = output_root / "capture_evidence.json"
        if evidence_path.exists() or evidence_path.is_symlink():
            raise CaptureError(f"capture evidence is create-only: {evidence_path}")
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return evidence
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr-p04-public-pool-capture-failure.v1",
            "task_id": TASK_ID,
            "status": "FAILED_PUBLIC_CORRECTION_VALIDATION_CAPTURE",
            "selection": {"path": str(selection_path), "sha256": selection_sha256},
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "device": str(device),
            "model_loaded": prefix is not None,
            "target_weights_accessed": False,
            "evaluation_truth_accessed": False,
            "phases": phases,
            "pool_counts": pool_counts,
            "peak_memory": helper._peak_memory(device) if helper is not None else {"host_max_rss_bytes": _max_rss_bytes()},
        }
        try:
            failure_path = output_root / "failure.json"
            if not failure_path.exists() and not failure_path.is_symlink():
                failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise
    finally:
        del prefix
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = run(args)
    except (CaptureError, OSError, RuntimeError, ValueError) as exc:
        print(f"P04 public capture failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "output_root": str(args.output_root.expanduser().resolve()),
                "correction_records": evidence["pools"]["correction"]["records"],
                "validation_records": evidence["pools"]["validation"]["records"],
                "model_loaded": evidence["access_contract"]["model_loaded"],
                "evaluation_truth_accessed": evidence["access_contract"]["evaluation_truth_accessed"],
                "peak_memory": evidence["peak_memory"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
