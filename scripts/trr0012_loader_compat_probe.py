"""Bounded synthetic compatibility probe for the portable TRR-0012 loader.

This probe validates only state schema/metadata/strict tensor restoration.  It
never loads the public embedding table, runs a decoder forward, opens truth, or
uses CUDA.  The synthetic state is removed after the probe; it is not a model
artifact or a deployment dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
from typing import Any

from safetensors.torch import save_file
import torch


EXPECTED_KEYS = frozenset(
    {
        "base.W",
        "base.b",
        "base.key.bias",
        "base.key.weight",
        "base.output.bias",
        "base.output.weight",
        "base.query.bias",
        "base.query.weight",
        "base.s",
        "base.value.bias",
        "base.value.weight",
        "down.bias",
        "down.weight",
        "up.bias",
        "up.weight",
    }
)
STATE_SCHEMA = "token-reconstruction.trr-p09-fixed-state.v1"
METHOD_ID = "continued_fixed_readout"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def run(*, package_root: Path, output: Path) -> dict[str, Any]:
    package_root = Path(package_root).expanduser().resolve()
    code_root = package_root / "code"
    if not code_root.is_dir() or (code_root / "trr0010_p09_fixed_loader.py").is_symlink():
        raise RuntimeError("portable package code root is unavailable")
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from token_reconstruction.trr0007_positionwise import build_residual_mlp512
    from trr0010_p09_fixed_loader import load_p09_fixed_state

    output = Path(output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"compatibility receipt is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    binding = _digest("TRR-0012 synthetic compatibility binding")
    with tempfile.TemporaryDirectory(prefix="trr0012-loader-compat-") as temporary:
        state_path = Path(temporary) / "synthetic_state.safetensors"
        model = build_residual_mlp512(
            hidden_size=2048,
            vocabulary_size=128256,
            context_width=128,
            bottleneck_size=512,
            seed=4005,
        )
        state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        if set(state) != EXPECTED_KEYS:
            raise RuntimeError("synthetic builder tensor keys differ from the loader contract")
        save_file(
            state,
            str(state_path),
            metadata={
                "schema": STATE_SCHEMA,
                "task_id": "TRR-0012",
                "method_id": METHOD_ID,
                "selected_step": "0",
                "bank_manifest_sha256": binding,
                "fit_manifest_sha256": binding,
                "schedule_semantic_sha256": binding,
                "base_state_sha256": binding,
                "embedding_sha256": binding,
                "runner_state_sha256": binding,
                "optimizer_state_external": "true",
                "serialization_only": "true",
            },
        )
        state_sha = _sha256(state_path)
        loaded = load_p09_fixed_state(
            state_path,
            expected_state_sha256=state_sha,
            expected_selected_step=0,
            expected_bank_manifest_sha256=binding,
            expected_fit_manifest_sha256=binding,
            expected_schedule_semantic_sha256=binding,
            expected_base_state_sha256=binding,
            expected_embedding_sha256=binding,
            expected_runner_state_sha256=binding,
        )
        if loaded.__class__.__name__ != "ResidualMLPPositionwiseDecoder":
            raise RuntimeError(f"unexpected restored class: {loaded.__class__.__name__}")
        if set(loaded.state_dict()) != EXPECTED_KEYS:
            raise RuntimeError("restored model tensor keys differ from the loader contract")
        state_bytes = state_path.stat().st_size

    result = {
        "schema": "token-reconstruction.trr0012-loader-compatibility.v1",
        "task_id": "TRR-0012",
        "status": "PASS_SYNTHETIC_STRICT_LOADER_COMPATIBILITY",
        "loader": "code/trr0010_p09_fixed_loader.py:load_p09_fixed_state",
        "state_schema": STATE_SCHEMA,
        "serialized_method_id": METHOD_ID,
        "restored_class": "ResidualMLPPositionwiseDecoder",
        "tensor_keys_sorted": sorted(EXPECTED_KEYS),
        "synthetic_state_bytes": state_bytes,
        "synthetic_state_sha256": state_sha,
        "synthetic_state_removed_after_probe": True,
        "public_readout_loaded": False,
        "full_vocabulary_forward": False,
        "cuda_used": False,
        "truth_opened": False,
        "source_text_loaded": False,
        "evaluation_truth_opened": False,
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["receipt"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(package_root=args.package_root, output=args.output), sort_keys=True))
    except Exception as exc:
        print(f"TRR0012 loader compatibility error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
