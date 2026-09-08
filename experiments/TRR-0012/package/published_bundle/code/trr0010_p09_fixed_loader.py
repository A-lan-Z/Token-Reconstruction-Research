"""Strict loader for the serialized TRR-P09 fixed decoder state.

TRR-P09 fixed-control checkpoints use the same 15 tensors as the published
TRR-0007 residual MLP512 decoder, but their state metadata has the P09 runner
schema.  The generic TRR-0007 loader intentionally rejects that schema.  This
small adapter binds the P09 file/hash and metadata, constructs the identical
residual decoder, and loads the tensors strictly for current-H inference.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch import nn

from token_reconstruction.trr0007_positionwise import build_residual_mlp512


P09_STATE_SCHEMA = "token-reconstruction.trr-p09-fixed-state.v1"
P09_METHOD_ID = "continued_fixed_readout"
RESIDUAL_BOTTLENECK_SIZE = 512
DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_VOCABULARY_SIZE = 128256
DEFAULT_CONTEXT_WIDTH = 128
DEFAULT_SEED = 4005
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_TENSOR_KEYS = frozenset(
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


class P09FixedLoaderError(ValueError):
    """Raised when a P09 fixed-state binding or tensor schema is invalid."""


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise P09FixedLoaderError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_metadata(metadata: Mapping[str, Any], *, name: str, expected: str) -> None:
    actual = metadata.get(name)
    if actual != expected:
        raise P09FixedLoaderError(
            f"P09 fixed-state metadata {name} differs: {actual!r} != {expected!r}"
        )


def load_p09_fixed_state(
    path: Path,
    *,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    bottleneck_size: int = RESIDUAL_BOTTLENECK_SIZE,
    seed: int = DEFAULT_SEED,
    expected_state_sha256: str,
    expected_selected_step: int,
    expected_bank_manifest_sha256: str,
    expected_fit_manifest_sha256: str | None = None,
    expected_schedule_semantic_sha256: str,
    expected_base_state_sha256: str,
    expected_embedding_sha256: str,
    expected_runner_state_sha256: str,
    expected_method_id: str = P09_METHOD_ID,
) -> nn.Module:
    """Load one hash- and metadata-bound P09 fixed residual decoder.

    The registration layer independently checks the file binding.  This
    adapter repeats the exact file hash and checkpoint metadata checks at the
    imported loader boundary, then requires the complete 15-tensor state set
    before strict loading into the identical TRR-0007 residual MLP512 class.
    """
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise P09FixedLoaderError(f"P09 fixed state is not a regular file: {path}")
    path = path.resolve()
    expected_state_sha256 = _require_digest(expected_state_sha256, name="expected_state_sha256")
    expected_bank_manifest_sha256 = _require_digest(expected_bank_manifest_sha256, name="expected_bank_manifest_sha256")
    expected_schedule_semantic_sha256 = _require_digest(expected_schedule_semantic_sha256, name="expected_schedule_semantic_sha256")
    expected_base_state_sha256 = _require_digest(expected_base_state_sha256, name="expected_base_state_sha256")
    expected_embedding_sha256 = _require_digest(expected_embedding_sha256, name="expected_embedding_sha256")
    expected_runner_state_sha256 = _require_digest(expected_runner_state_sha256, name="expected_runner_state_sha256")
    if expected_fit_manifest_sha256 is not None:
        expected_fit_manifest_sha256 = _require_digest(expected_fit_manifest_sha256, name="expected_fit_manifest_sha256")
    if expected_method_id != P09_METHOD_ID:
        raise P09FixedLoaderError(f"P09 fixed method identity changed: {expected_method_id!r}")
    if isinstance(expected_selected_step, bool) or int(expected_selected_step) < 0:
        raise P09FixedLoaderError("expected_selected_step must be a non-negative integer")
    if int(hidden_size) != DEFAULT_HIDDEN_SIZE or int(vocabulary_size) != DEFAULT_VOCABULARY_SIZE:
        raise P09FixedLoaderError("P09 fixed decoder hidden/vocabulary geometry is frozen")
    if int(context_width) != DEFAULT_CONTEXT_WIDTH or int(bottleneck_size) != RESIDUAL_BOTTLENECK_SIZE:
        raise P09FixedLoaderError("P09 fixed decoder context/bottleneck geometry is frozen")

    actual_state_sha256 = _digest_file(path)
    if actual_state_sha256 != expected_state_sha256:
        raise P09FixedLoaderError(
            f"P09 fixed state file hash differs: {actual_state_sha256} != {expected_state_sha256}"
        )
    try:
        state = load_file(str(path), device="cpu")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            serialized_keys = frozenset(handle.keys())
    except Exception as exc:
        raise P09FixedLoaderError(f"cannot read P09 fixed state: {path}") from exc
    if serialized_keys != _EXPECTED_TENSOR_KEYS or set(state) != set(_EXPECTED_TENSOR_KEYS):
        missing = sorted(_EXPECTED_TENSOR_KEYS - serialized_keys)
        extra = sorted(serialized_keys - _EXPECTED_TENSOR_KEYS)
        raise P09FixedLoaderError(f"P09 fixed tensor keys differ; missing={missing}, extra={extra}")

    _require_metadata(metadata, name="schema", expected=P09_STATE_SCHEMA)
    _require_metadata(metadata, name="method_id", expected=P09_METHOD_ID)
    _require_metadata(metadata, name="selected_step", expected=str(int(expected_selected_step)))
    _require_metadata(metadata, name="bank_manifest_sha256", expected=expected_bank_manifest_sha256)
    _require_metadata(metadata, name="fit_manifest_sha256", expected=expected_fit_manifest_sha256 or expected_bank_manifest_sha256)
    _require_metadata(metadata, name="schedule_semantic_sha256", expected=expected_schedule_semantic_sha256)
    _require_metadata(metadata, name="base_state_sha256", expected=expected_base_state_sha256)
    _require_metadata(metadata, name="embedding_sha256", expected=expected_embedding_sha256)
    _require_metadata(metadata, name="runner_state_sha256", expected=expected_runner_state_sha256)
    _require_metadata(metadata, name="optimizer_state_external", expected="true")
    _require_metadata(metadata, name="serialization_only", expected="true")

    model = build_residual_mlp512(
        hidden_size=DEFAULT_HIDDEN_SIZE,
        vocabulary_size=DEFAULT_VOCABULARY_SIZE,
        context_width=DEFAULT_CONTEXT_WIDTH,
        bottleneck_size=RESIDUAL_BOTTLENECK_SIZE,
        seed=int(seed),
    )
    expected_model_keys = set(model.state_dict())
    if expected_model_keys != set(_EXPECTED_TENSOR_KEYS):
        raise P09FixedLoaderError("TRR-0007 residual model tensor key contract changed")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise P09FixedLoaderError("P09 fixed tensors do not match residual MLP512 geometry") from exc
    model.eval()
    # Keep the validated header available to runner evidence without making it
    # part of the model state or changing inference semantics.
    model._trr_p09_state_metadata = dict(metadata)  # type: ignore[attr-defined]
    model._trr_p09_state_sha256 = actual_state_sha256  # type: ignore[attr-defined]
    return model


__all__ = [
    "P09FixedLoaderError",
    "P09_METHOD_ID",
    "P09_STATE_SCHEMA",
    "load_p09_fixed_state",
]
