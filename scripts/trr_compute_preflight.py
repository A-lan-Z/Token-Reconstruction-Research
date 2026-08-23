#!/usr/bin/env python3
"""Fail closed when a planned CUDA run lacks a reasonable safety margin."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-free-gib", type=float, required=True)
    parser.add_argument("--probe-mib", type=int, required=True)
    args = parser.parse_args()
    if args.minimum_free_gib <= 0 or args.probe_mib <= 0:
        raise RuntimeError("CUDA preflight limits must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda")
    free_before, total = torch.cuda.mem_get_info(device)
    required = int(args.minimum_free_gib * 1024**3)
    if free_before < required:
        raise RuntimeError(
            f"CUDA free memory {free_before} is below required {required}"
        )
    probe_bytes = args.probe_mib * 1024**2
    probe = torch.empty(probe_bytes, dtype=torch.uint8, device=device)
    probe.zero_()
    torch.cuda.synchronize(device)
    del probe
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    print(
        json.dumps(
            {
                "status": "CUDA_PREFLIGHT_PASS",
                "checked_utc": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "device": torch.cuda.get_device_name(device),
                "total_bytes": total,
                "free_bytes_before": free_before,
                "free_bytes_after": free_after,
                "minimum_free_bytes": required,
                "probe_bytes": probe_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
