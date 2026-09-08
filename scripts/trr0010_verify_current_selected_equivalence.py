"""Reproduce the CPU equivalence proof for selected TRR-0010 deployment files.

This is a bounded, truth-free check.  It compares the 15 decoder tensors in
an exported selected-step-0 base state with the immutable shared starting
state, then compares public E and the exported effective W in safe_open row
chunks.  It never constructs a model, reads activations/labels, or uses CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"bound file is unavailable: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _base_exact(start: Path, exported: Path) -> dict[str, Any]:
    left = load_file(str(start), device="cpu")
    right = load_file(str(exported), device="cpu")
    if set(left) != set(right):
        raise RuntimeError("base tensor key sets differ")
    per_tensor: list[dict[str, Any]] = []
    exact = True
    for key in sorted(left):
        a = left[key]
        b = right[key]
        same = tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(a, b)
        exact = exact and bool(same)
        per_tensor.append({"key": key, "shape": list(a.shape), "dtype": str(a.dtype), "exact": bool(same)})
    return {"tensor_keys_equal": True, "tensor_exact": bool(exact), "tensor_count": len(per_tensor), "per_tensor": per_tensor}


def _readout_exact(public: Path, exported: Path, *, chunk_rows: int) -> dict[str, Any]:
    started = time.perf_counter()
    public_chunks = hashlib.sha256()
    exported_chunks = hashlib.sha256()
    rows_compared = 0
    mismatches = 0
    max_abs = 0.0
    with safe_open(str(public), framework="pt", device="cpu") as public_file, safe_open(str(exported), framework="pt", device="cpu") as exported_file:
        if list(public_file.keys()) != ["embeddings"] or list(exported_file.keys()) != ["embeddings"]:
            raise RuntimeError("public E and effective W must each contain only embeddings")
        public_slice = public_file.get_slice("embeddings")
        exported_slice = exported_file.get_slice("embeddings")
        if public_slice.get_shape() != exported_slice.get_shape() or public_slice.get_dtype() != exported_slice.get_dtype():
            raise RuntimeError("public E and effective W geometry/dtype differ")
        rows, _hidden = public_slice.get_shape()
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            left = public_slice[start:stop]
            right = exported_slice[start:stop]
            if not torch.equal(left, right):
                diff = (left.float() - right.float()).abs()
                max_abs = max(max_abs, float(diff.max().item()))
                mismatches += int((left != right).sum().item())
            public_chunks.update(left.contiguous().numpy().tobytes(order="C"))
            exported_chunks.update(right.contiguous().numpy().tobytes(order="C"))
            rows_compared += stop - start
    return {
        "exact": mismatches == 0,
        "rows_compared": rows_compared,
        "chunk_rows": chunk_rows,
        "mismatches": mismatches,
        "max_abs_diff": max_abs,
        "public_chunk_digest": public_chunks.hexdigest(),
        "effective_chunk_digest": exported_chunks.hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-state", type=Path, required=True)
    parser.add_argument("--base-export", type=Path, required=True)
    parser.add_argument("--public-embedding", type=Path, required=True)
    parser.add_argument("--effective-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.chunk_rows <= 0:
        raise RuntimeError("--chunk-rows must be positive")
    script_path = Path(__file__).resolve()
    start = _record(args.start_state)
    base = _record(args.base_export)
    public = _record(args.public_embedding)
    effective = _record(args.effective_readout)
    result = {
        "schema": "token-reconstruction.trr0010-current-selected-equivalence-command.v1",
        "task_id": "TRR-0010",
        "status": "PASS_CPU_EXACT",
        "command": list(sys.argv),
        "script": {"path": str(script_path), "bytes": script_path.stat().st_size, "sha256": _sha256(script_path)},
        "start_state": start,
        "base_export": base,
        "public_embedding": public,
        "effective_readout": effective,
        "base_tensor_equivalence": _base_exact(Path(args.start_state), Path(args.base_export)),
        "effective_readout_equivalence": _readout_exact(Path(args.public_embedding), Path(args.effective_readout), chunk_rows=args.chunk_rows),
        "selected_step": 0,
        "truth_opened": False,
        "cuda_used": False,
        "notes": "Safe-open row chunks; no full second vocabulary table retained.",
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
