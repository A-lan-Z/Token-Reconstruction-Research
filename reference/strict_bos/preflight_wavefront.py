#!/usr/bin/env python3
"""Synthetic semantic and fail-closed checks for the Round 003 decoders."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


IMPLEMENTATION = Path(__file__).resolve().parent
if str(IMPLEMENTATION) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION))

import round001_teacher as teacher  # noqa: E402
import round003_wavefront as wavefront  # noqa: E402


class FakeCache:
    def __init__(self) -> None:
        self.batch: int | None = None
        self.length = 0

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.batch is None or indices.ndim != 1:
            raise RuntimeError("invalid fake-cache select")
        if indices.numel() and int(indices.max()) >= self.batch:
            raise RuntimeError("fake-cache select is out of range")
        self.batch = int(indices.numel())

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.batch is None or repeats <= 0:
            raise RuntimeError("invalid fake-cache repeat")
        self.batch *= repeats


class FakePrecut:
    def __init__(self) -> None:
        self.forward_batches: list[int] = []

    def new_cache(self) -> FakeCache:
        return FakeCache()

    def run_cached(
        self, input_ids: torch.Tensor, cache: FakeCache, start_pos: int
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] <= 0:
            raise RuntimeError("fake precut input changed")
        if cache.length != start_pos:
            raise RuntimeError("fake cache length changed")
        batch, tokens = input_ids.shape
        self.forward_batches.append(int(batch))
        if cache.batch is None:
            cache.batch = batch
        elif cache.batch != batch:
            raise RuntimeError("fake cache batch changed")
        hidden = torch.zeros(batch, tokens, 2048, dtype=torch.float32)
        index = input_ids.remainder(512).to(torch.long).unsqueeze(-1)
        hidden.scatter_(2, index, 1.0)
        cache.length += tokens
        return hidden


def passive_fixture() -> tuple[list[teacher.PassiveRow], torch.Tensor, torch.Tensor]:
    mask = torch.zeros(128, 128, dtype=torch.long)
    mask[:, 0] = 1
    mask[0, 1] = 1
    mask[1, 1] = 1
    position = mask.cumsum(dim=-1).sub(1).clamp_min(0)
    activation = torch.zeros(128, 128, 2048, dtype=torch.float32)
    activation[:, 0, teacher.BOS_TOKEN_ID % 512] = 1.0
    activation[0, 1, 0] = 1.0
    activation[1, 1, 7] = 1.0
    rows = [
        teacher.PassiveRow(
            row_index=index,
            activation=activation[index],
            attention_mask=mask[index],
            position_ids=position[index],
        )
        for index in range(128)
    ]
    for row in rows:
        row.validate()
    candidates = torch.arange(512, dtype=torch.int32).view(1, 1, 512).expand(
        128, 128, 512
    ).clone()
    confidence = torch.zeros(128, 128, dtype=torch.float32)
    confidence[0, 1] = 1.0
    return rows, candidates, confidence


def chunk_fixture() -> tuple[list[teacher.PassiveRow], torch.Tensor, torch.Tensor]:
    mask = torch.zeros(128, 128, dtype=torch.long)
    mask[:, :2] = 1
    position = mask.cumsum(dim=-1).sub(1).clamp_min(0)
    activation = torch.zeros(128, 128, 2048, dtype=torch.float32)
    activation[:, 0, teacher.BOS_TOKEN_ID % 512] = 1.0
    activation[:, 1, 7] = 1.0
    rows = [
        teacher.PassiveRow(
            row_index=index,
            activation=activation[index],
            attention_mask=mask[index],
            position_ids=position[index],
        )
        for index in range(128)
    ]
    for row in rows:
        row.validate()
    candidates = torch.arange(512, dtype=torch.int32).view(1, 1, 512).expand(
        128, 128, 512
    ).clone()
    confidence = torch.zeros(128, 128, dtype=torch.float32)
    return rows, candidates, confidence


def expect_failure(callback: Any, description: str) -> str:
    try:
        callback()
    except Exception as exc:  # noqa: BLE001 - deliberate fail-closed harness
        return f"{description}: {type(exc).__name__}: {exc}"
    raise AssertionError(f"synthetic mismatch was accepted: {description}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if wavefront.MAX_CANDIDATE_SEQUENCES_PER_FORWARD != 384:
        raise AssertionError("frozen candidate-forward limit changed")
    rows, candidates, confidence = passive_fixture()
    precut = FakePrecut()
    reference = wavefront.decode_row_serial_source(
        rows,
        candidates=candidates,
        a1_confidence=confidence,
        precut=precut,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    batched = wavefront.decode_wavefront_source(
        rows,
        candidates=candidates,
        a1_confidence=confidence,
        precut=FakePrecut(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    comparison = wavefront.compare_outputs(reference, batched)

    chunk_rows, chunk_candidates, chunk_confidence = chunk_fixture()
    chunk_reference = wavefront.decode_row_serial_source(
        chunk_rows,
        candidates=chunk_candidates,
        a1_confidence=chunk_confidence,
        precut=FakePrecut(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    chunk_precut = FakePrecut()
    chunk_batched = wavefront.decode_wavefront_source(
        chunk_rows,
        candidates=chunk_candidates,
        a1_confidence=chunk_confidence,
        precut=chunk_precut,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    chunk_comparison = wavefront.compare_outputs(
        chunk_reference, chunk_batched
    )
    if max(chunk_precut.forward_batches) != 384:
        raise AssertionError("chunked candidate batch limit changed")
    if chunk_batched.candidate_simulations != 128 * 32:
        raise AssertionError("chunked candidate simulation count changed")
    if int(chunk_batched.token_ids[127, 1]) != 7:
        raise AssertionError("chunked final-parent winner changed")

    if int(reference.route_codes[0, 1]) != teacher.ROUTE_A1:
        raise AssertionError("synthetic fast route changed")
    if int(reference.route_codes[1, 1]) != teacher.ROUTE_A2_K32:
        raise AssertionError("synthetic A2 route changed")
    if int(reference.token_ids[1, 1]) != 7:
        raise AssertionError("synthetic A2 winner changed")

    wrong_token_values = {
        name: value.clone() for name, value in batched.tensors().items()
    }
    wrong_token_values["token_ids"][1, 1] = 8
    wrong_token = wavefront.output_from_tensors(wrong_token_values)

    wrong_float_values = {
        name: value.clone() for name, value in batched.tensors().items()
    }
    wrong_float_values["tier_score_margin"][1, 1, 0] += 2e-5
    wrong_float = wavefront.output_from_tensors(wrong_float_values)

    wrong_nonfinite_values = {
        name: value.clone() for name, value in batched.tensors().items()
    }
    wrong_nonfinite_values["tier_score_margin"][0, 1, 0] = float("inf")

    failures = [
        expect_failure(
            lambda: wavefront.compare_outputs(reference, wrong_token),
            "integer mismatch",
        ),
        expect_failure(
            lambda: wavefront.compare_outputs(reference, wrong_float),
            "finite tolerance mismatch",
        ),
        expect_failure(
            lambda: wavefront.output_from_tensors(wrong_nonfinite_values),
            "non-finite category mismatch",
        ),
    ]
    stacked_a = wavefront.stack_outputs([51, 52], [reference, batched])
    stacked_b = wavefront.stack_outputs([51, 52], [batched, reference])
    byte_counts_a = {
        name: value.numel() * value.element_size() for name, value in stacked_a.items()
    }
    byte_counts_b = {
        name: value.numel() * value.element_size() for name, value in stacked_b.items()
    }
    if byte_counts_a != byte_counts_b:
        raise AssertionError("equal operational schemas have unequal tensor bytes")

    payload = {
        "schema": "ersoy.adaptive_a1_a2_goal.round003.synthetic_wavefront_preflight.v2",
        "wavefront_code_sha256": hashlib.sha256(
            Path(wavefront.__file__).read_bytes()
        ).hexdigest(),
        "candidate_sequence_limit": wavefront.MAX_CANDIDATE_SEQUENCES_PER_FORWARD,
        "row_serial_wavefront_comparison": comparison,
        "chunked_128_parent_comparison": chunk_comparison,
        "chunked_max_candidate_sequences_per_forward": max(
            chunk_precut.forward_batches
        ),
        "chunked_candidate_simulations": chunk_batched.candidate_simulations,
        "fast_route": int(reference.route_codes[0, 1]),
        "a2_route": int(reference.route_codes[1, 1]),
        "a2_winner": int(reference.token_ids[1, 1]),
        "negative_tests": failures,
        "operational_tensor_bytes": sum(byte_counts_a.values()),
        "operational_schema_byte_counts_exact": True,
        "checks_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
