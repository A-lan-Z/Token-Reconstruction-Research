#!/usr/bin/env python3
"""Summarize paired P04 activation drift without opening truth or source rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors.torch import load_file

SCHEMA = "token-reconstruction.trr-p04-paired-activation-drift.v1"
TASK_ID = "TRR-P04"
CONDITIONS = ("public_base", "p04_evaluator_target_update_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cell_summary(base: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    active = mask.reshape(-1)
    av = base.reshape(-1, base.shape[-1])[active]
    bv = target.reshape(-1, target.shape[-1])[active]
    diff = bv - av
    base_rms = torch.sqrt(torch.mean(av * av))
    diff_rms = torch.sqrt(torch.mean(diff * diff))
    cosine = (av * bv).sum(dim=1) / (
        torch.linalg.vector_norm(av, dim=1) * torch.linalg.vector_norm(bv, dim=1)
    ).clamp_min(1e-12)
    return {
        "active_vectors": int(av.shape[0]),
        "difference_rms": float(diff_rms),
        "relative_difference_rms_to_base": float(diff_rms / base_rms.clamp_min(1e-12)),
        "mean_vector_cosine": float(torch.mean(cosine)),
        "minimum_vector_cosine": float(torch.min(cosine)),
    }


def run(input_root: Path, output_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    started_utc = utc_now()
    input_root = input_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError(f"refusing to overwrite create-only output: {output_path}")
    index_path = input_root / "observation_index.json"
    base_path = input_root / "observations" / "public_base.safetensors"
    target_path = input_root / "observations" / "p04_evaluator_target_update_v1.safetensors"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    base = load_file(str(base_path), device="cpu")
    target = load_file(str(target_path), device="cpu")
    for key in ("activations", "attention_mask", "position_ids"):
        if key not in base or key not in target:
            raise RuntimeError(f"missing paired tensor: {key}")
        if tuple(base[key].shape) != tuple(target[key].shape):
            raise RuntimeError(f"paired tensor geometry mismatch: {key}")
    if not torch.equal(base["attention_mask"], target["attention_mask"]):
        raise RuntimeError("paired attention masks differ")
    if not torch.equal(base["position_ids"], target["position_ids"]):
        raise RuntimeError("paired position IDs differ")

    base_activations = base["activations"].float()
    target_activations = target["activations"].float()
    active_mask = base["attention_mask"].to(torch.bool)
    overall = cell_summary(base_activations, target_activations, active_mask)
    expanded = active_mask.unsqueeze(-1).expand_as(base_activations)
    av = base_activations[expanded].reshape(-1, base_activations.shape[-1])
    bv = target_activations[expanded].reshape(-1, target_activations.shape[-1])
    diff = bv - av
    base_rms = torch.sqrt(torch.mean(av * av))
    target_rms = torch.sqrt(torch.mean(bv * bv))
    diff_rms = torch.sqrt(torch.mean(diff * diff))
    cosine = (av * bv).sum(dim=1) / (
        torch.linalg.vector_norm(av, dim=1) * torch.linalg.vector_norm(bv, dim=1)
    ).clamp_min(1e-12)

    value: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS_PAIRED_DRIFT_NO_TRUTH",
        "created_utc": utc_now(),
        "selection": {
            key: index["selection"].get(key)
            for key in ("path", "sha256", "record_count", "anchor_count", "record_order_sha256", "anchor_order_sha256")
        },
        "conditions": list(CONDITIONS),
        "observation_geometry": list(base_activations.shape),
        "active_vectors": int(av.shape[0]),
        "metrics": {
            "activation_rms_base": float(base_rms),
            "activation_rms_target": float(target_rms),
            "difference_rms": float(diff_rms),
            "difference_mean_abs": float(torch.mean(torch.abs(diff))),
            "difference_max_abs": float(torch.max(torch.abs(diff))),
            "relative_difference_rms_to_base": float(diff_rms / base_rms.clamp_min(1e-12)),
            "mean_vector_cosine": float(torch.mean(cosine)),
            "minimum_vector_cosine": float(torch.min(cosine)),
            "fraction_exact_equal_vectors": float(torch.mean(torch.all(diff == 0, dim=1).float())),
        },
        "cell_metrics": {},
        "access": {
            "evaluation_truth_opened": False,
            "source_text_or_tokens_read": False,
            "student_states_loaded": False,
            "target_update_weights_read": False,
            "observation_payloads_read": True,
            "selection_metadata_read": True,
        },
        "inputs": {
            "public_base_observation": {"path": str(base_path), "bytes": base_path.stat().st_size, "sha256": sha256_file(base_path)},
            "target_observation": {"path": str(target_path), "bytes": target_path.stat().st_size, "sha256": sha256_file(target_path)},
            "observation_index": {"path": str(index_path), "bytes": index_path.stat().st_size, "sha256": sha256_file(index_path)},
        },
        "execution": {
            "argv": list(sys.argv),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cpu",
            "started_utc": started_utc,
            "elapsed_seconds": None,
            "evaluation_truth_opened": False,
        },
    }
    records = index["records"]
    for style in sorted({str(record["style"]) for record in records}):
        for length in sorted({int(record["length_stratum"]) for record in records}):
            ordinals = [
                ordinal
                for ordinal, record in enumerate(records)
                if str(record["style"]) == style and int(record["length_stratum"]) == length
            ]
            if not ordinals:
                continue
            cell_mask = active_mask[ordinals]
            cell = cell_summary(base_activations[ordinals], target_activations[ordinals], cell_mask)
            cell["records"] = len(ordinals)
            value["cell_metrics"][f"{style}|L{length}"] = cell
    value["execution"]["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.input_root, args.output)
    print(json.dumps({"status": value["status"], "output": str(args.output.resolve()), "active_vectors": value["active_vectors"], "cell_count": len(value["cell_metrics"]), "elapsed_seconds": value["execution"]["elapsed_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
