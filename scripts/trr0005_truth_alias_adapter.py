#!/usr/bin/env python3
"""Run the frozen TRR-0005 truth CLI with an alias-safe save boundary.

Provenance: the frozen ``scripts/trr0005_produce_confirmation.py`` at the
held TRR-0005 execution commit (da82f6cac45e09ae83452198344c547553cb4433).
Its ``prepare_truth`` function reuses each domain's labels, attention mask,
and position tensors for the two public conditions.  The installed safetensors
writer rejects those shared-storage entries.  This adapter leaves the producer
CLI and all source/record logic unchanged, and patches only the producer
module's ``save_file`` symbol while invoking ``producer.main(["truth", ...])``.  Every tensor is detached, made contiguous,
and cloned immediately before serialization; metadata and string tensor keys
are passed through unchanged.  It does not load evaluator truth or alter the
selection, panel, or truth-binding rules.

Canonical invocation (producer-only preparation after selection/panel freeze;
evaluator opening still waits for full prediction freeze):

    PYTHONPATH=src:scripts .venv-trr0005/bin/python \
      scripts/trr0005_truth_alias_adapter.py truth <frozen-truth-arguments>
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import sys
from typing import Any

import torch
from safetensors.torch import save_file as _save_file

try:  # Repository test/import form keeps one shared producer module.
    from scripts import trr0005_produce_confirmation as _producer
except ModuleNotFoundError:  # Direct script execution with PYTHONPATH=src:scripts.
    import trr0005_produce_confirmation as _producer


class TruthAliasAdapterError(ValueError):
    """Raised when the frozen truth writer receives an invalid tensor map."""


def clone_tensors_for_save(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Break shared storage while retaining tensor keys, values, and dtypes."""

    if not isinstance(tensors, Mapping):
        raise TruthAliasAdapterError("truth tensor payload must be a mapping")
    cloned: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if not isinstance(key, str) or not key:
            raise TruthAliasAdapterError("truth tensor keys must be nonempty strings")
        if key in cloned:
            raise TruthAliasAdapterError(f"duplicate truth tensor key: {key}")
        if not isinstance(tensor, torch.Tensor):
            raise TruthAliasAdapterError(f"truth tensor is not a torch.Tensor: {key}")
        # contiguous() makes the boundary acceptable for safetensors; clone()
        # then guarantees that no two serialized entries share storage.
        cloned[key] = tensor.detach().contiguous().clone()
    return cloned


def save_file_alias_safe(
    tensors: Mapping[str, torch.Tensor],
    filename: str,
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Serialize a cloned tensor map using the original safetensors writer."""

    cloned = clone_tensors_for_save(tensors)
    preserved_metadata = dict(metadata) if metadata is not None else None
    _save_file(cloned, filename, metadata=preserved_metadata)


def run_truth_cli(argv: Sequence[str]) -> int:
    """Invoke the frozen producer CLI with the alias-safe save boundary."""

    arguments = list(argv)
    if not arguments or arguments[0] != "truth":
        raise TruthAliasAdapterError("adapter accepts the frozen producer 'truth' command only")
    original_save_file = _producer.save_file
    _producer.save_file = save_file_alias_safe
    try:
        return int(_producer.main(arguments))
    finally:
        _producer.save_file = original_save_file


def main(argv: list[str] | None = None) -> int:
    return run_truth_cli(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
