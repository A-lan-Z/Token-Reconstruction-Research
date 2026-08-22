#!/usr/bin/env python3
"""Create a small, dependency-free deterministic TRR bootstrap artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


FORMAT_VERSION = "trr.bootstrap-smoke.v1"
ALGORITHM = "splitmix64-v1"
UINT64_MASK = (1 << 64) - 1
SAMPLE_COUNT = 8


def uint64_seed(value: str) -> int:
    """Parse a base-10 seed in the unsigned 64-bit range."""

    try:
        seed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be a base-10 integer") from exc
    if not 0 <= seed <= UINT64_MASK:
        raise argparse.ArgumentTypeError(
            f"seed must be between 0 and {UINT64_MASK}, inclusive"
        )
    return seed


def splitmix64(state: int) -> tuple[int, int]:
    """Advance SplitMix64 and return the new state and output word."""

    state = (state + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    value ^= value >> 31
    return state, value & UINT64_MASK


def build_payload(seed: int) -> dict[str, object]:
    """Return the complete deterministic data payload for one seed."""

    if isinstance(seed, bool) or not 0 <= seed <= UINT64_MASK:
        raise ValueError("seed is outside the unsigned 64-bit range")

    state = seed
    words: list[int] = []
    for _ in range(SAMPLE_COUNT):
        state, value = splitmix64(state)
        words.append(value)

    xor_value = 0
    for value in words:
        xor_value ^= value
    sum_value = sum(words) & UINT64_MASK

    return {
        "algorithm": ALGORITHM,
        "format_version": FORMAT_VERSION,
        "sample_count": SAMPLE_COUNT,
        "samples": [
            {
                "hex": f"0x{value:016x}",
                "index": index,
                "uint64": value,
            }
            for index, value in enumerate(words)
        ],
        "seed": seed,
        "summary": {
            "sum_mod_2_64_hex": f"0x{sum_value:016x}",
            "xor_hex": f"0x{xor_value:016x}",
        },
    }


def write_payload(payload: dict[str, object], output: Path) -> None:
    """Serialize payload with stable ordering, whitespace, encoding, and newline."""

    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=uint64_seed)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        write_payload(build_payload(args.seed), args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: error: unable to write {args.output}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
