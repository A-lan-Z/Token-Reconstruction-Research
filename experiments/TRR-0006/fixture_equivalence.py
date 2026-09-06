#!/usr/bin/env python3
"""Bounded TRR-0006 frozen-pair fixture check.

This utility is deliberately separate from the TRR-0005 128-record runner.
It reads only public ``activations``, ``attention_mask``, and ``position_ids``
from the retained observation files.  The raw files also contain token IDs,
but this program never requests that tensor.  ``input`` checks the captured
192-token prefix against the retained 128-token view.  ``cuda`` then uses the
same decoder loader, CPU-to-CUDA FP32 conversion, full normalized E table, and
argmax path as the retained predictor, comparing raw-prefix, trimmed, and
saved prediction IDs exactly for one or two rows per cell.

No selection, fitting, source text, target labels, truth, candidate arrays, or
timing comparison is performed here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file


CELL_ORDER = (
    "finance__public_base",
    "finance__public_lora_2601",
    "pile__public_base",
    "pile__public_lora_2601",
)
METHODS = (
    "enriched__affine_causal_h_attention128",
    "enriched__affine_trained_diagonal_attention128",
)
STATE_RELATIVE = {
    METHODS[0]: Path(
        "experiments/TRR-0005/joint_fit_qknorm_v1/enriched/"
        "affine_causal_h_attention128/selected.safetensors"
    ),
    METHODS[1]: Path(
        "experiments/TRR-0005/joint_fit_v1/enriched/"
        "affine_trained_diagonal_attention128/selected.safetensors"
    ),
}
STATE_SHA256 = {
    METHODS[0]: "ee910b14ad6f282bb933ea44ad24453cb5cce1470c65dbc09d8bcc16f3e8abfd",
    METHODS[1]: "696eb9fc951e85356a06575faf18a2011616692a086bdac3b2fa368e69d599a2",
}
RAW_SHA256 = {
    "finance__public_base": "5adebb05b5c1f7411f75a9e5d0ccb91fa46bf64799542950eca0f846aa714691",
    "finance__public_lora_2601": "b6f6fe342505d6f92a3e7156e93617b6496b403e3444daf5c3549464ee4aeeac",
    "pile__public_base": "c9925ac106ace70ca1608ef78ab652bfb64932001ae1a15d4a2efbae48629aeb",
    "pile__public_lora_2601": "c32e1f4dab12a03d7d1773011bc0d2715e1586f180364a0f71644b42e7655341",
}
TRIM_SHA256 = {
    "finance__public_base": "da033e05362cf4731d7132c066777102cc2569d5ff2119785bd6086ce9fe8eb9",
    "finance__public_lora_2601": "d2fdda52f9276a8e3864334f05b36f3bc431e8828428deb0abcf273dfba67dc6",
    "pile__public_base": "8faea0d80e52fe1a56827f249aaddc8205cfcd88b24a5ca9a3d6030b7c8d7e64",
    "pile__public_lora_2601": "864ef938da0bd5991cf6b016f560a2320df6252beb09c455d17132ea144281b3",
}
PREDICTION_RELATIVE = "experiments/TRR-0005/fresh_confirmation_v1/predictions_v2_contract_export"
EMBEDDING_RELATIVE = "outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors"
EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
PREFIX_TOKENS = 128
CAPTURE_TOKENS = 192
BOS_TOKEN_ID = 128000
MIN_FREE_GIB = 8.0
MAX_RESERVED_GIB = 6.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(path: Path) -> Path:
    value = path.expanduser().resolve()
    if value.is_symlink() or not value.is_dir():
        raise RuntimeError(f"TRR-0005 root is unavailable: {value}")
    return value


def _paths(trr5: Path, cell: str) -> tuple[Path, Path, Path]:
    style, condition = cell.split("__", 1)
    raw = trr5 / "outputs/TRR-0005/fresh_confirmation_capture_v2" / f"{cell}.padded192.safetensors"
    trimmed = trr5 / "experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations" / f"{cell}.safetensors"
    prediction_root = trr5 / PREDICTION_RELATIVE / style / condition
    return raw, trimmed, prediction_root


def _header(path: Path) -> dict[str, Any]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {"keys": sorted(handle.keys()), "metadata": dict(handle.metadata() or {})}


def _load_public(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load the three allowed tensors; intentionally never request token_ids."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        required = {"activations", "attention_mask", "position_ids"}
        if not required.issubset(set(handle.keys())):
            raise RuntimeError(f"public observation lacks required tensors: {path}")
        activations = handle.get_tensor("activations").contiguous()
        mask = handle.get_tensor("attention_mask").contiguous()
        positions = handle.get_tensor("position_ids").contiguous()
        metadata = dict(handle.metadata() or {})
    if activations.dtype != torch.bfloat16 or tuple(activations.shape[-1:]) != (HIDDEN_SIZE,):
        raise RuntimeError(f"unexpected public activation geometry/dtype: {path}")
    if tuple(mask.shape) != tuple(activations.shape[:2]) or tuple(positions.shape) != tuple(mask.shape):
        raise RuntimeError(f"public observation sidecar geometry changed: {path}")
    return activations, mask.to(torch.bool), positions, metadata


def _input_check(trr5: Path, records: int) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for cell in CELL_ORDER:
        raw_path, trimmed_path, _ = _paths(trr5, cell)
        raw_hash = sha256_file(raw_path)
        trim_hash = sha256_file(trimmed_path)
        if raw_hash != RAW_SHA256[cell] or trim_hash != TRIM_SHA256[cell]:
            raise RuntimeError(f"retained public input hash changed for {cell}")
        raw_h, raw_mask, raw_pos, raw_meta = _load_public(raw_path)
        trim_h, trim_mask, trim_pos, trim_meta = _load_public(trimmed_path)
        if tuple(raw_h.shape) != (128, CAPTURE_TOKENS, HIDDEN_SIZE):
            raise RuntimeError(f"raw capture geometry changed for {cell}: {tuple(raw_h.shape)}")
        if tuple(trim_h.shape) != (128, PREFIX_TOKENS, HIDDEN_SIZE):
            raise RuntimeError(f"trimmed capture geometry changed for {cell}: {tuple(trim_h.shape)}")
        raw_prefix_h = raw_h[:records, :PREFIX_TOKENS]
        raw_prefix_mask = raw_mask[:records, :PREFIX_TOKENS]
        raw_prefix_pos = raw_pos[:records, :PREFIX_TOKENS]
        trim_h = trim_h[:records]
        trim_mask = trim_mask[:records]
        trim_pos = trim_pos[:records]
        cells[cell] = {
            "raw_path": str(raw_path),
            "raw_bytes": raw_path.stat().st_size,
            "raw_sha256": raw_hash,
            "trimmed_path": str(trimmed_path),
            "trimmed_bytes": trimmed_path.stat().st_size,
            "trimmed_sha256": trim_hash,
            "records_checked": records,
            "raw_shape": [128, CAPTURE_TOKENS, HIDDEN_SIZE],
            "trimmed_shape": [128, PREFIX_TOKENS, HIDDEN_SIZE],
            "activation_prefix_exact": bool(torch.equal(raw_prefix_h, trim_h)),
            "mask_prefix_exact": bool(torch.equal(raw_prefix_mask, trim_mask)),
            "position_prefix_exact": bool(torch.equal(raw_prefix_pos, trim_pos)),
            "raw_metadata": {k: raw_meta.get(k) for k in ("capture_batch_records", "capture_sequence_tokens", "public_full_forward", "target_truth_accessed")},
            "trimmed_metadata": {k: trim_meta.get(k) for k in ("batch_records", "capture_sequence_tokens", "public_full_forward", "target_truth_accessed")},
        }
        if not all(cells[cell][key] for key in ("activation_prefix_exact", "mask_prefix_exact", "position_prefix_exact")):
            raise RuntimeError(f"raw/trimmed public prefix differs for {cell}")
    return {"records": records, "cells": cells, "truth_opened": False}


def _prediction_row(path: Path, method: str, cell: str, records: int) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if "predictions" not in set(handle.keys()):
            raise RuntimeError(f"saved prediction tensor is absent: {path}")
        prediction = handle.get_tensor("predictions").to(torch.long).contiguous()
        metadata = dict(handle.metadata() or {})
    if metadata.get("task_id") != "TRR-0005" or metadata.get("cell_id") != cell or metadata.get("method_id") != method:
        raise RuntimeError(f"saved prediction binding changed: {path}")
    if tuple(prediction.shape) != (128, PREFIX_TOKENS):
        raise RuntimeError(f"saved prediction geometry changed: {path}")
    if not prediction[:records, 0].eq(BOS_TOKEN_ID).all().item():
        raise RuntimeError(f"saved prediction BOS changed: {path}")
    return prediction[:records]


@torch.inference_mode()
def _predict(model: torch.nn.Module, embeddings: torch.Tensor, activation: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    # This is the retained _JointAdapter boundary: CPU BF16 H -> device FP32,
    # mask -> device bool, full E, model forward, argmax, IDs back to CPU.
    # The retained runner then applies _normalize_prediction: padding is -1
    # and position zero is the known BOS ID, irrespective of its raw argmax.
    h = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
    valid = mask.to(device=device, dtype=torch.bool).unsqueeze(0)
    logits = model(h, valid, embeddings)
    ids = logits.argmax(dim=-1)[0].to(device="cpu", dtype=torch.long).contiguous()
    active = mask.to(device="cpu", dtype=torch.bool)
    output = torch.full_like(ids, -1)
    output[active] = ids[active]
    output[0] = BOS_TOKEN_ID
    return output


def _cuda_gate(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("qualified CUDA device is unavailable")
    free, total = torch.cuda.mem_get_info(device)
    reserved = int(torch.cuda.memory_reserved(device))
    minimum = int(MIN_FREE_GIB * 2**30)
    maximum = int(MAX_RESERVED_GIB * 2**30)
    if int(free) < minimum or reserved > maximum:
        raise RuntimeError(f"CUDA resource gate failed: free={free}, reserved={reserved}")
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "free_bytes_before_load": int(free),
        "total_bytes": int(total),
        "reserved_bytes_before_load": reserved,
        "minimum_free_bytes": minimum,
        "maximum_reserved_bytes": maximum,
        "host_max_rss_bytes_before_load": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }


def _cuda_check(trr5: Path, embedding: Path, records: int, device: torch.device) -> dict[str, Any]:
    gate = _cuda_gate(device)
    if sha256_file(embedding) != EMBEDDING_SHA256:
        raise RuntimeError("shared normalized E hash changed")
    # Match the retained loader: read CPU F32 E, then copy the unchanged table
    # to CUDA.  No normalization or reduced vocabulary is permitted here.
    table = load_file(str(embedding), device="cpu")
    if set(table) != {"embeddings"} or tuple(table["embeddings"].shape) != (VOCAB_SIZE, HIDDEN_SIZE) or table["embeddings"].dtype != torch.float32:
        raise RuntimeError("shared normalized E geometry/dtype changed")
    embeddings = table["embeddings"].contiguous().to(device=device).contiguous()
    torch.cuda.synchronize(device)
    del table
    gc.collect()
    from token_reconstruction.trr0005_joint_decoder import load_decoder_state

    results: dict[str, Any] = {}
    for method in METHODS:
        state_path = trr5 / STATE_RELATIVE[method]
        if sha256_file(state_path) != STATE_SHA256[method]:
            raise RuntimeError(f"selected state hash changed for {method}")
        base_method = method.split("__", 1)[1]
        model = load_decoder_state(
            state_path,
            method_id=base_method,
            hidden_size=HIDDEN_SIZE,
            vocabulary_size=VOCAB_SIZE,
            context_width=128,
        ).to(device=device).eval()
        model.requires_grad_(False)
        method_cells: dict[str, Any] = {}
        for cell in CELL_ORDER:
            raw_path, trimmed_path, prediction_dir = _paths(trr5, cell)
            raw_h, raw_mask, raw_pos, _ = _load_public(raw_path)
            trim_h, trim_mask, trim_pos, _ = _load_public(trimmed_path)
            del raw_pos, trim_pos
            saved_path = prediction_dir / f"{method}.safetensors"
            saved = _prediction_row(saved_path, method, cell, records)
            rows: list[dict[str, Any]] = []
            for row in range(records):
                raw_ids = _predict(model, embeddings, raw_h[row], raw_mask[row], device)
                trim_ids = _predict(model, embeddings, trim_h[row], trim_mask[row], device)
                raw_prefix = raw_ids[:PREFIX_TOKENS]
                raw_trim_exact = bool(torch.equal(raw_prefix, trim_ids))
                saved_exact = bool(torch.equal(trim_ids, saved[row]))
                rows.append({
                    "record_index": row,
                    "raw_output_tokens": int(raw_ids.shape[0]),
                    "trimmed_output_tokens": int(trim_ids.shape[0]),
                    "raw_prefix_equals_trimmed": raw_trim_exact,
                    "trimmed_equals_saved": saved_exact,
                    "raw_prefix_equals_saved": bool(torch.equal(raw_prefix, saved[row])),
                })
                if not (raw_trim_exact and saved_exact and rows[-1]["raw_prefix_equals_saved"]):
                    raise RuntimeError(f"strict fixture ID equality failed for {cell}/{method}/row{row}")
            method_cells[cell] = {"prediction_path": str(saved_path), "records": rows}
        results[method] = {
            "state_path": str(state_path),
            "state_sha256": STATE_SHA256[method],
            "attention_mode": model.attention_mode,
            "attention_score_mode": model.attention_score_mode,
            "cells": method_cells,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
        if int(free) < int(MIN_FREE_GIB * 2**30):
            raise RuntimeError(f"CUDA post-method free-memory gate failed: {free}")
    return {
        "records": records,
        "resource_gate": gate,
        "embedding_path": str(embedding),
        "embedding_sha256": EMBEDDING_SHA256,
        "methods": results,
        "truth_opened": False,
        "timing_comparison": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("headers", "input", "cuda"), default="headers")
    parser.add_argument("--trr5-root", type=Path, required=True)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--records", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trr5 = _root(args.trr5_root)
    if args.mode == "headers":
        payload: dict[str, Any] = {"mode": "headers", "cells": {}}
        for cell in CELL_ORDER:
            raw, trimmed, _ = _paths(trr5, cell)
            payload["cells"][cell] = {"raw": _header(raw), "trimmed": _header(trimmed)}
        payload["truth_opened"] = False
    elif args.mode == "input":
        payload = {"mode": "input_prefix_only", **_input_check(trr5, args.records)}
        payload["inference_equivalence_proven"] = False
    else:
        if args.embedding is None:
            args.embedding = trr5.parent.parent / EMBEDDING_RELATIVE
        embedding = args.embedding.expanduser().resolve()
        if embedding.is_symlink() or not embedding.is_file():
            raise RuntimeError(f"normalized public E is unavailable: {embedding}")
        if "src" not in sys.path:
            sys.path.insert(0, str(trr5 / "src"))
        payload = {"mode": "cuda_id_equivalence", **_cuda_check(trr5, embedding, args.records, torch.device(args.device))}
        payload["inference_equivalence_proven"] = True
    payload["task_id"] = "TRR-0006"
    payload["selected_pair_only"] = True
    payload["source_text_loaded"] = False
    payload["target_labels_loaded"] = False
    payload["candidate_arrays_persisted"] = False
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"output is create-only and already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
