"""CPU-only strict verification of the selected expanded deployment export.

The check uses public fitting assets only.  It verifies that the selected
base-only deployment state contains exactly the decoder tensors from the
selected directional checkpoint and that the serialized full-vocabulary W is
exactly FP32(E + Delta) at every vocabulary row, with the supported rows
scattered by their serialized support IDs.  No activations, labels, source
text, capture, or evaluation truth are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _write_create_only(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from trr0010_model import load_positionwise_model_state

    torch.set_num_threads(1)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    base_path = Path(args.base).expanduser().resolve()
    effective_w = Path(args.effective_w).expanduser().resolve()
    embedding = Path(args.embedding).expanduser().resolve()
    binding_path = Path(args.binding).expanduser().resolve()
    for path, label in (
        (checkpoint, "selected checkpoint"),
        (base_path, "base-only deployment state"),
        (effective_w, "effective readout W"),
        (embedding, "public embedding E"),
        (binding_path, "production binding"),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{label} is not a regular file: {path}")

    binding = _json(binding_path, label="production binding")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("public_embedding"), dict):
        raise RuntimeError("production binding lacks public_embedding artifact")
    embedding_binding = artifacts["public_embedding"]
    expected_e_sha = embedding_binding.get("sha256")
    expected_e_bytes = int(embedding_binding.get("bytes"))
    if Path(str(embedding_binding.get("path"))).expanduser().resolve() != embedding:
        raise RuntimeError("public embedding path differs from the bound production artifact")
    actual_e_bytes = embedding.stat().st_size
    actual_e_sha = _sha256_file(embedding)
    if actual_e_bytes != expected_e_bytes or actual_e_sha != expected_e_sha:
        raise RuntimeError("public embedding bytes/SHA differs from the bound production artifact")

    checkpoint_sha = _sha256_file(checkpoint)
    base_sha = _sha256_file(base_path)
    w_sha = _sha256_file(effective_w)
    checkpoint_bytes = checkpoint.stat().st_size
    base_bytes = base_path.stat().st_size
    w_bytes = effective_w.stat().st_size

    # Strictly load the deployment state through the registered native loader.
    decoder = load_positionwise_model_state(
        base_path,
        method_id="trr0007_residual_mlp512",
        hidden_size=2048,
        vocabulary_size=128256,
        context_width=128,
        bottleneck_size=512,
    )
    decoder.eval()
    if decoder.training:
        raise RuntimeError("strict base-only loader did not leave the decoder in eval mode")

    with safe_open(str(checkpoint), framework="pt", device="cpu") as checkpoint_file:
        checkpoint_meta = dict(checkpoint_file.metadata() or {})
        checkpoint_keys = set(checkpoint_file.keys())
        checkpoint_base = {key: checkpoint_file.get_tensor(key) for key in checkpoint_keys if key.startswith("base.")}
        support_ids = checkpoint_file.get_tensor("support_ids").to(dtype=torch.int64)
        delta_rows = checkpoint_file.get_tensor("delta_rows")
    with safe_open(str(base_path), framework="pt", device="cpu") as base_file:
        base_meta = dict(base_file.metadata() or {})
        base_keys = set(base_file.keys())
        base_tensors = {key: base_file.get_tensor(key) for key in base_keys}
    expected_base_keys = {"base." + key: value for key, value in base_tensors.items()}
    if set(expected_base_keys) != set(checkpoint_base):
        raise RuntimeError("selected base export/checkpoint base tensor key sets differ")
    base_mismatches = [
        key for key in sorted(expected_base_keys)
        if not torch.equal(expected_base_keys[key], checkpoint_base[key])
    ]
    if base_mismatches:
        raise RuntimeError(f"selected base export differs from checkpoint base tensors: {base_mismatches[:5]}")

    if tuple(support_ids.shape) != (45631,) or tuple(delta_rows.shape) != (45631, 2048):
        raise RuntimeError("selected support/delta geometry differs from expanded B1")
    if support_ids.numel() != torch.unique(support_ids).numel():
        raise RuntimeError("selected support IDs are not unique")
    if int(support_ids.min()) < 0 or int(support_ids.max()) >= 128256:
        raise RuntimeError("selected support IDs are outside the vocabulary")

    with safe_open(str(embedding), framework="pt", device="cpu") as e_file, safe_open(str(effective_w), framework="pt", device="cpu") as w_file:
        e_meta = dict(e_file.metadata() or {})
        w_meta = dict(w_file.metadata() or {})
        if list(e_file.keys()) != ["embeddings"] or list(w_file.keys()) != ["embeddings"]:
            raise RuntimeError("E/W serialized tensor key differs from embeddings")
        e_shape = tuple(e_file.get_slice("embeddings").get_shape())
        w_shape = tuple(w_file.get_slice("embeddings").get_shape())
        if e_shape != (128256, 2048) or w_shape != e_shape:
            raise RuntimeError(f"E/W shape differs: E={e_shape}, W={w_shape}")
        if e_meta.get("dtype") not in (None, "torch.float32", "float32"):
            raise RuntimeError(f"public E metadata dtype differs: {e_meta.get('dtype')}")
        if w_meta.get("selected_step") != "12000" or w_meta.get("selected_checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("W metadata does not bind selected step/checkpoint")
        chunk_rows = int(args.chunk_rows)
        mismatch_rows = 0
        max_abs_diff = 0.0
        compared_rows = 0
        for start in range(0, 128256, chunk_rows):
            stop = min(start + chunk_rows, 128256)
            e_chunk = e_file.get_slice("embeddings")[start:stop].to(dtype=torch.float32)
            w_chunk = w_file.get_slice("embeddings")[start:stop].to(dtype=torch.float32)
            expected = e_chunk.clone()
            mask = (support_ids >= start) & (support_ids < stop)
            if bool(mask.any()):
                local_ids = (support_ids[mask] - start).to(dtype=torch.long)
                expected[local_ids] = expected[local_ids] + delta_rows[mask]
            differences = (expected - w_chunk).abs()
            chunk_max = float(differences.max().item()) if differences.numel() else 0.0
            max_abs_diff = max(max_abs_diff, chunk_max)
            mismatch_rows += int((differences.amax(dim=1) != 0).sum().item())
            compared_rows += stop - start
        if mismatch_rows or max_abs_diff != 0.0:
            raise RuntimeError(f"serialized W differs from exact FP32(E+Delta): rows={mismatch_rows}, max_abs_diff={max_abs_diff}")

    return {
        "schema": "token-reconstruction.trr0010-expanded-selected-export-cpu-verification.v1",
        "task_id": "TRR-0010",
        "status": "PASS_CPU_STRICT_BASE_AND_EXACT_EFFECTIVE_W",
        "command": list(sys.argv),
        "truth_opened": False,
        "source_text_loaded": False,
        "activations_loaded": False,
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint_bytes, "sha256": checkpoint_sha, "metadata": checkpoint_meta},
        "base_decoder_state": {"path": str(base_path), "bytes": base_bytes, "sha256": base_sha, "metadata": base_meta, "strict_loader_eval": True, "base_tensor_keys_exact": True, "base_tensor_count": len(base_tensors)},
        "public_embedding": {"path": str(embedding), "bytes": actual_e_bytes, "sha256": actual_e_sha, "metadata": e_meta, "binding_sha256": expected_e_sha},
        "effective_readout_w": {"path": str(effective_w), "bytes": w_bytes, "sha256": w_sha, "metadata": w_meta},
        "exact_effective_w_check": {"formula": "serialized_W == FP32(public_E + serialized_delta_rows at support_ids)", "support_count": int(support_ids.numel()), "rows_compared": compared_rows, "chunk_rows": int(args.chunk_rows), "mismatches": 0, "max_abs_diff": 0.0, "tensor_exact": True},
        "source_bindings": {
            "model_loader": {"path": str(Path("scripts/trr0010_model.py").resolve()), "sha256": _sha256_file(Path("scripts/trr0010_model.py").resolve())},
            "verification_script": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--effective-w", required=True, type=Path)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        result = run(args)
    except Exception as exc:
        failure = {"schema": "token-reconstruction.trr0010-expanded-selected-export-cpu-verification.v1", "task_id": "TRR-0010", "status": "FAIL_CPU_EXPORT_VERIFICATION", "command": list(sys.argv), "error_type": type(exc).__name__, "error": str(exc), "truth_opened": False, "source_text_loaded": False, "activations_loaded": False}
        _write_create_only(output, failure)
        raise
    _write_create_only(output, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
