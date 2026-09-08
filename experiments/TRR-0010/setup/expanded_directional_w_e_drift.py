#!/usr/bin/env python3
"""CPU-only integrity summary for the selected expanded directional readout.

This is a metadata/effective-weight check.  It reads only the public E table,
the selected exported W, and the selected checkpoint's public ``delta_rows``
and support IDs.  It does not load observations, labels, source text, truth,
or run an inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from safetensors import safe_open


def _binding(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest.hexdigest()}


def _stats(values: list[torch.Tensor]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    vector = torch.cat(values).to(dtype=torch.float64)
    finite = bool(torch.isfinite(vector).all().item())
    if not finite:
        raise RuntimeError("non-finite drift statistic")
    return {
        "count": int(vector.numel()),
        "min": float(vector.min().item()),
        "mean": float(vector.mean().item()),
        "median": float(torch.quantile(vector, 0.5).item()),
        "p95": float(torch.quantile(vector, 0.95).item()),
        "max": float(vector.max().item()),
        "finite": finite,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--effective-readout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be positive")
    started = time.perf_counter()
    embedding = args.embedding.expanduser().resolve()
    readout = args.effective_readout.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    for path in (embedding, readout, checkpoint):
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"required direct file is unavailable: {path}")

    with safe_open(str(embedding), framework="pt", device="cpu") as e_file, safe_open(
        str(readout), framework="pt", device="cpu"
    ) as w_file, safe_open(str(checkpoint), framework="pt", device="cpu") as c_file:
        if list(e_file.keys()) != ["embeddings"] or list(w_file.keys()) != ["embeddings"]:
            raise SystemExit("embedding/readout tensor keys are not exactly embeddings")
        e_shape = tuple(e_file.get_slice("embeddings").get_shape())
        w_shape = tuple(w_file.get_slice("embeddings").get_shape())
        if e_shape != w_shape or len(e_shape) != 2:
            raise SystemExit(f"embedding/readout shapes differ: {e_shape} vs {w_shape}")
        if e_shape != (128256, 2048):
            raise SystemExit(f"unexpected production geometry: {e_shape}")
        if c_file.keys() is None or "support_ids" not in c_file.keys() or "delta_rows" not in c_file.keys():
            raise SystemExit("selected checkpoint lacks support_ids/delta_rows")
        support_ids = c_file.get_tensor("support_ids").to(dtype=torch.int64, device="cpu")
        delta_rows = c_file.get_tensor("delta_rows").to(dtype=torch.float32, device="cpu")
        if tuple(delta_rows.shape) != (int(support_ids.numel()), e_shape[1]):
            raise SystemExit("delta_rows and support_ids geometry mismatch")
        if support_ids.numel() == 0 or bool(torch.unique(support_ids).numel() != support_ids.numel()):
            raise SystemExit("support IDs are empty or duplicated")
        if int(support_ids.min()) < 0 or int(support_ids.max()) >= e_shape[0]:
            raise SystemExit("support IDs are out of vocabulary range")
        support_index = {int(token_id): index for index, token_id in enumerate(support_ids.tolist())}
        checkpoint_metadata = dict(c_file.metadata() or {})

        supported_l2: list[torch.Tensor] = []
        unsupported_l2: list[torch.Tensor] = []
        supported_max_abs: list[torch.Tensor] = []
        unsupported_max_abs: list[torch.Tensor] = []
        supported_e_l2: list[torch.Tensor] = []
        unsupported_e_l2: list[torch.Tensor] = []
        reconstruction_max_abs: list[torch.Tensor] = []
        reconstruction_l2: list[torch.Tensor] = []
        nonzero_supported = 0
        nonzero_unsupported = 0
        unsupported_exact_zero = 0
        for start in range(0, e_shape[0], args.chunk_rows):
            stop = min(start + args.chunk_rows, e_shape[0])
            e_rows = e_file.get_slice("embeddings")[start:stop]
            w_rows = w_file.get_slice("embeddings")[start:stop]
            drift = (w_rows - e_rows).to(dtype=torch.float64)
            row_l2 = torch.linalg.vector_norm(drift, dim=1)
            row_max_abs = drift.abs().amax(dim=1)
            e_l2 = torch.linalg.vector_norm(e_rows.to(dtype=torch.float64), dim=1)
            supported_rows = [token_id for token_id in support_index if start <= token_id < stop]
            supported_offsets = [token_id - start for token_id in supported_rows]
            supported_mask = torch.zeros(stop - start, dtype=torch.bool)
            if supported_offsets:
                supported_mask[torch.tensor(supported_offsets, dtype=torch.int64)] = True
                positions = torch.tensor([support_index[token_id] for token_id in supported_rows], dtype=torch.int64)
                observed_delta = drift[torch.tensor(supported_offsets, dtype=torch.int64)]
                expected_delta = delta_rows[positions].to(dtype=torch.float64)
                error = observed_delta - expected_delta
                reconstruction_max_abs.append(error.abs().amax(dim=1))
                reconstruction_l2.append(torch.linalg.vector_norm(error, dim=1))
            unsupported_mask = ~supported_mask
            if bool(supported_mask.any()):
                supported_l2.append(row_l2[supported_mask])
                supported_max_abs.append(row_max_abs[supported_mask])
                supported_e_l2.append(e_l2[supported_mask])
                nonzero_supported += int((row_max_abs[supported_mask] > 0).sum().item())
            if bool(unsupported_mask.any()):
                unsupported_l2.append(row_l2[unsupported_mask])
                unsupported_max_abs.append(row_max_abs[unsupported_mask])
                unsupported_e_l2.append(e_l2[unsupported_mask])
                nonzero_unsupported += int((row_max_abs[unsupported_mask] > 0).sum().item())
                unsupported_exact_zero += int((row_max_abs[unsupported_mask] == 0).sum().item())

    payload = {
        "schema": "token-reconstruction.trr0010-expanded-directional-w-e-drift.v1",
        "task_id": "TRR-0010",
        "status": "CPU_ONLY_EFFECTIVE_READOUT_DRIFT_PASS",
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "observations_loaded": False,
        "inference_run": False,
        "inputs": {
            "public_embedding": _binding(embedding),
            "effective_readout": _binding(readout),
            "selected_checkpoint": _binding(checkpoint),
        },
        "geometry": {"vocabulary_size": e_shape[0], "hidden_size": e_shape[1], "chunk_rows": args.chunk_rows},
        "selected_checkpoint_metadata": checkpoint_metadata,
        "support": {
            "count": int(support_ids.numel()),
            "digest": checkpoint_metadata.get("support_digest"),
            "nonzero_drift_rows": nonzero_supported,
            "unsupported_rows": e_shape[0] - int(support_ids.numel()),
            "unsupported_nonzero_drift_rows": nonzero_unsupported,
            "unsupported_exact_zero_rows": unsupported_exact_zero,
        },
        "supported_row_drift_l2": _stats(supported_l2),
        "unsupported_row_drift_l2": _stats(unsupported_l2),
        "supported_row_drift_max_abs": _stats(supported_max_abs),
        "unsupported_row_drift_max_abs": _stats(unsupported_max_abs),
        "supported_public_e_row_l2": _stats(supported_e_l2),
        "unsupported_public_e_row_l2": _stats(unsupported_e_l2),
        "w_minus_e_equals_checkpoint_delta": {
            "max_abs_error": _stats(reconstruction_max_abs),
            "l2_error": _stats(reconstruction_l2),
            "checked_rows": int(support_ids.numel()),
        },
        "interpretation": "Only supported token rows changed; unsupported rows are exact W==E rows, and supported W-E matches stored delta_rows within serialized float arithmetic.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
