#!/usr/bin/env python3
"""Diagnose public-fit token-type transfer for selected Track B states.

This is a public-data diagnostic.  It uses only the public inverse-train
labels/activations and the disjoint public validation labels/activations.  It
does not load the shared panel or any evaluator-private sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
from safetensors.torch import load_file

from token_reconstruction.experiment_runtime import synchronize, utc_now
from token_reconstruction.inverse import load_inverse
from token_reconstruction.standalone_decoder import (
    angular_prediction_tensor,
    load_token_decoder,
    prediction_tensor,
    validate_embedding_table,
)


ROOT = Path(__file__).resolve().parents[3]
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
METHODS = (
    "angular_inverse_control",
    "tied_affine_token_ce",
    "residual_mlp256_token_ce",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _load(path: Path, key: str) -> torch.Tensor:
    state = load_file(path.resolve(), device="cpu")
    if set(state) != {key}:
        raise RuntimeError(f"{path} must contain exactly {key!r}")
    return state[key].contiguous()


def _flat_x(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[2] != HIDDEN_SIZE or value.shape[1] <= 1:
        raise RuntimeError("activation geometry changed")
    return value[:, 1:, :].reshape(-1, HIDDEN_SIZE).contiguous()


def _flat_y(value: torch.Tensor, records: int, positions: int) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (records, positions):
        raise RuntimeError("label geometry changed")
    if value[:, 0].ne(128000).any().item():
        raise RuntimeError("public labels BOS changed")
    if value[:, 1:].lt(0).any().item() or value[:, 1:].ge(VOCAB_SIZE).any().item():
        raise RuntimeError("public labels token range changed")
    return value[:, 1:].reshape(-1).long().contiguous()


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--validation-observations", type=Path, required=True)
    parser.add_argument("--validation-truth", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"output must be create-only: {args.output}")
    if args.batch_size <= 0:
        raise RuntimeError("batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started_utc = utc_now()
    input_root = args.input_root.resolve()
    fit_obs_path = input_root / "fit_observations.safetensors"
    fit_truth_path = input_root / "fit_truth.safetensors"
    fit_obs = _load(fit_obs_path, "activations")
    fit_truth = _load(fit_truth_path, "token_ids")
    fit_x = _flat_x(fit_obs)
    fit_y = _flat_y(fit_truth, int(fit_obs.shape[0]), int(fit_obs.shape[1]))

    val_obs_path = args.validation_observations.resolve()
    val_truth_path = args.validation_truth.resolve()
    val_obs = _load(val_obs_path, "activations")
    val_truth = _load(val_truth_path, "token_ids")
    val_x = _flat_x(val_obs)
    val_y = _flat_y(val_truth, int(val_obs.shape[0]), int(val_obs.shape[1]))

    embedding_path = args.embedding_table.resolve()
    embeddings = _load(embedding_path, "embeddings").float().contiguous()
    validate_embedding_table(embeddings, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)
    fit_types = set(int(value) for value in fit_y.tolist())
    seen = torch.tensor([int(value) in fit_types for value in val_y.tolist()], dtype=torch.bool)
    unseen = ~seen
    overlap_types = set(int(value) for value in val_y.tolist()) & fit_types
    fit_only_types = fit_types - set(int(value) for value in val_y.tolist())
    val_types = set(int(value) for value in val_y.tolist())

    embedding_runtime = embeddings.to(device=device, dtype=torch.float32)
    results: dict[str, Any] = {}
    for method in METHODS:
        state_path = args.fit_root.resolve() / f"{method}.safetensors"
        if method == "angular_inverse_control":
            model = load_inverse(state_path, hidden_size=HIDDEN_SIZE, device=device)
        else:
            model = load_token_decoder(
                state_path,
                method_id=method,
                hidden_size=HIDDEN_SIZE,
                vocab_size=VOCAB_SIZE,
                device=device,
                logit_scale=16.0,
                bottleneck_size=256,
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize()
        started = time.perf_counter()
        if method == "angular_inverse_control":
            fit_pred = angular_prediction_tensor(
                model, fit_x, embedding_runtime, device=device, batch_size=args.batch_size
            ).long()
            val_pred = angular_prediction_tensor(
                model, val_x, embedding_runtime, device=device, batch_size=args.batch_size
            ).long()
        else:
            fit_pred = prediction_tensor(
                model, fit_x, embedding_runtime, device=device, batch_size=args.batch_size
            ).long()
            val_pred = prediction_tensor(
                model, val_x, embedding_runtime, device=device, batch_size=args.batch_size
            ).long()
        synchronize()
        elapsed = time.perf_counter() - started
        val_correct = val_pred.eq(val_y)
        fit_correct = fit_pred.eq(fit_y)
        def rate(mask: torch.Tensor) -> dict[str, Any]:
            count = int(mask.sum().item())
            correct = int((val_correct & mask).sum().item())
            return {"positions": count, "correct": correct, "accuracy": correct / count if count else None}
        results[method] = {
            "state": _file(state_path),
            "fit": {
                "positions": int(fit_y.numel()),
                "correct": int(fit_correct.sum().item()),
                "accuracy": float(fit_correct.float().mean().item()),
            },
            "validation": {
                "all": rate(torch.ones_like(seen)),
                "fit_type_seen": rate(seen),
                "fit_type_unseen": rate(unseen),
            },
            "prediction_seconds": elapsed,
            "peak_memory": {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None,
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None,
            },
        }
        del model, fit_pred, val_pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "token-reconstruction.trr0003-track-b-token-transfer-diagnostic.v1",
        "task_id": "TRR-0003",
        "track": "track_b",
        "status": "PUBLIC_LABEL_TRANSFER_DIAGNOSTIC",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "git_commit": _commit(),
        "code": {"script": _file(Path(__file__).resolve())},
        "model": {
            "id": "meta-llama/Llama-3.2-1B-Instruct",
            "revision": "9213176726f574b556790deb65791e0c5aa438b6",
            "cut_depth": 4,
        },
        "inputs": {
            "fit_observations": _file(fit_obs_path),
            "fit_truth": _file(fit_truth_path),
            "validation_observations": _file(val_obs_path),
            "validation_truth": _file(val_truth_path),
            "embedding_table": _file(embedding_path),
            "fit_root": str(args.fit_root.resolve()),
        },
        "token_type_overlap": {
            "fit_positions": int(fit_y.numel()),
            "validation_positions": int(val_y.numel()),
            "fit_unique_types": len(fit_types),
            "validation_unique_types": len(val_types),
            "overlap_unique_types": len(overlap_types),
            "validation_types_seen_in_fit": len(overlap_types) / len(val_types) if val_types else None,
            "validation_positions_seen_in_fit": int(seen.sum().item()),
            "validation_positions_unseen_in_fit": int(unseen.sum().item()),
            "validation_position_seen_fraction": float(seen.float().mean().item()),
            "fit_only_unique_types": len(fit_only_types),
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "embedding_transfer_once": True,
            "candidate_simulations": 0,
            "public_prefix_calls": 0,
            "panel_loaded": False,
            "private_sidecar_loaded": False,
        },
        "methods": results,
        "notes": [
            "Seen/unseen means whether the validation token ID appeared anywhere in public fit labels.",
            "This diagnostic uses public labels only and does not open the shared panel or private truth.",
            "The selected decoder states were chosen by the separate public-validation amendment.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
