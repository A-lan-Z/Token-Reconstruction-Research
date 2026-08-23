#!/usr/bin/env python3
"""Compare the generalized historical policy with the pinned native decoder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch

from token_reconstruction.a1a2_configuration_search import (
    ROUTE_FAST_A1,
    ROUTE_TIER,
    decode_policy,
    historical_anchor_spec,
    resolve_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_fixture(preflight: Any, rows: list[Any], candidates: torch.Tensor, confidence: torch.Tensor) -> dict[str, Any]:
    native = preflight.wavefront.decode_wavefront_source(
        rows,
        candidates=candidates,
        a1_confidence=confidence,
        precut=preflight.FakePrecut(),
        device=torch.device("cpu"),
    )
    observations = torch.stack([row.activation for row in rows]).contiguous()
    mask = torch.stack([row.attention_mask for row in rows]).to(torch.long).contiguous()
    positions = torch.stack([row.position_ids for row in rows]).to(torch.long).contiguous()
    generalized = decode_policy(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        a1_confidence=confidence,
        precut=preflight.FakePrecut(),
        device=torch.device("cpu"),
        policy=resolve_policy(historical_anchor_spec(), {}),
    )
    prediction_mismatches = int(generalized.predictions.ne(native.token_ids.to(torch.long)).sum().item())
    if prediction_mismatches:
        raise RuntimeError("generalized historical predictions differ from native")
    if generalized.candidate_simulations != native.candidate_simulations:
        raise RuntimeError("generalized historical simulation count differs from native")
    native_fast = native.route_codes.eq(preflight.teacher.ROUTE_A1)
    native_tier = (
        native.route_codes.eq(preflight.teacher.ROUTE_A2_K32)
        | native.route_codes.eq(preflight.teacher.ROUTE_A2_K128)
        | native.route_codes.eq(preflight.teacher.ROUTE_A2_K512)
    )
    fast_mismatches = int(generalized.routes.eq(ROUTE_FAST_A1).ne(native_fast).sum().item())
    tier_mismatches = int(generalized.routes.eq(ROUTE_TIER).ne(native_tier).sum().item())
    if fast_mismatches or tier_mismatches:
        raise RuntimeError("generalized historical route categories differ from native")
    return {
        "prediction_mismatches": prediction_mismatches,
        "fast_route_mismatches": fast_mismatches,
        "tier_route_mismatches": tier_mismatches,
        "native_candidate_simulations": native.candidate_simulations,
        "generalized_candidate_simulations": generalized.candidate_simulations,
        "native_covered_positions": native.covered_positions,
        "generalized_covered_positions": int(generalized.predictions.ge(0).sum().item()),
    }


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve(strict=True)
    preflight_path = root / "reference" / "strict_bos" / "preflight_wavefront.py"
    preflight = import_path("trr0002_r1_native_preflight", preflight_path)
    passive = compare_fixture(preflight, *preflight.passive_fixture())
    chunk = compare_fixture(preflight, *preflight.chunk_fixture())
    payload = {
        "schema": "token-reconstruction.trr0002-owner-r1-historical-common-runner-preflight.v1",
        "status": "PASS",
        "policy": resolve_policy(historical_anchor_spec(), {}).serialized(),
        "passive_fixture": passive,
        "chunk_fixture": chunk,
        "native_wavefront_source_sha256": hashlib.sha256(
            Path(preflight.wavefront.__file__).read_bytes()
        ).hexdigest(),
        "generalized_source_sha256": hashlib.sha256(
            (root / "src" / "token_reconstruction" / "a1a2_configuration_search.py").read_bytes()
        ).hexdigest(),
    }
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
