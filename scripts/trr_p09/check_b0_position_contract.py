#!/usr/bin/env python3
"""Check the published B0 attention/position padding convention only.

This deliberately requests only ``attention_mask`` and ``position_ids`` from
B0.  It never opens token IDs, H activations, source text, or truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

B0_ROWS = 1200
WIDTH = 192
B0_SHA256 = "a55814759dfa9d2567587935063fc49e44d8bff949c50014793deb982ebdf35d"

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def run(path: Path, output: Path) -> dict:
    path = path.expanduser().resolve()
    actual = digest(path)
    if actual != B0_SHA256:
        raise RuntimeError("B0 payload hash changed")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        # Do not request token_ids or activations; this is a mask/position check.
        mask = handle.get_tensor("attention_mask").contiguous()
        positions = handle.get_tensor("position_ids").contiguous()
    if tuple(mask.shape) != (B0_ROWS, WIDTH) or tuple(positions.shape) != (B0_ROWS, WIDTH):
        raise RuntimeError("B0 mask/position geometry changed")
    active_arange = True
    all_arange = True
    inactive_values: set[int] = set()
    for row in range(B0_ROWS):
        active = int(mask[row].sum().item())
        expected_active = torch.arange(active, dtype=positions.dtype)
        active_arange &= bool(torch.equal(positions[row, :active], expected_active))
        all_arange &= bool(torch.equal(positions[row], torch.arange(WIDTH, dtype=positions.dtype)))
        inactive_values.update(int(value) for value in positions[row, active:].tolist())
    result = {
        "schema": "token-reconstruction.trr-p09-b0-position-contract.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "payload": {"path": str(path), "bytes": path.stat().st_size, "sha256": actual},
        "requested_tensor_keys": ["attention_mask", "position_ids"],
        "forbidden_tensor_keys_requested": ["activations", "token_ids"],
        "geometry": {"records": B0_ROWS, "sequence_tokens": WIDTH},
        "observed": {"active_arange": active_arange, "all_arange": all_arange, "inactive_position_values": sorted(inactive_values)},
        "model_loaded": False,
        "source_text_loaded": False,
        "truth_opened": False,
        "h_activations_loaded": False,
    }
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.b0, args.output), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
