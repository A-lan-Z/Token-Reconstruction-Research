"""Faithful standalone bridge for the retained historical InputLens/A1.

The historical public A1 artifact is a frozen affine map followed by cosine
scoring against the normalized public input-embedding table.  This module
reproduces that inference path without fitting, calibration, candidate
simulation, or public-prefix calls.  It is deliberately separate from the
new standalone decoders so the retained artifact can be tested as a faithful
implementation before architecture or training comparisons are made.

The source implementation applies ``activation @ W.T + b`` in float32,
normalizes that projected vector, casts it to the normalized embedding table's
dtype for the matrix product, and multiplies the resulting logits by
``exp(s)``.  The embedding table is prepared once with float32 row
normalization, matching ``round001_teacher.normalize_public_embeddings``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


HISTORICAL_INPUTLENS_SCHEMA = "token-reconstruction.trr0004-historical-inputlens-bridge.v1"
HISTORICAL_LENS_STATE_DOMAIN = b"ersoy-a1-lens-state-v1"
HISTORICAL_HIDDEN_SIZE = 2048
HISTORICAL_HIDDEN_MARKER = 0
HISTORICAL_CORPUS = "alpaca"


class HistoricalInputLensError(RuntimeError):
    """Raised when the retained historical bridge contract is violated."""


@dataclass(frozen=True)
class HistoricalInputLensSpec:
    """Fixed semantics of the retained checkpoint, kept explicit in evidence."""

    hidden_size: int = HISTORICAL_HIDDEN_SIZE
    hidden_marker: int = HISTORICAL_HIDDEN_MARKER
    corpus: str = HISTORICAL_CORPUS
    projection: str = "activation.float32 @ W.float32.T + b.float32"
    projection_normalization: str = "torch.nn.functional.normalize(projected, dim=-1, eps=1e-12)"
    embedding_preparation: str = "F.normalize(embedding.detach().float32, dim=-1), nan_to_num"
    embedding_runtime_cast: str = "projected_normalized.to(normalized_embeddings.dtype)"
    logit_scale: str = "torch.exp(s.float32)"
    output_dtype: str = "torch.float32"
    fitted_parameters: str = "none at inference; W, b, s are loaded from the retained checkpoint"
    candidate_simulations: int = 0
    public_prefix_calls: int = 0


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HistoricalInputLensError(f"{label} must be a regular file: {path}")
    return path


def file_sha256(path: Path) -> str:
    """Hash a checkpoint or input resource without changing its bytes."""

    _regular_file(path, label="resource")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Return the historical state digest used by prior task evidence."""

    digest = hashlib.sha256(HISTORICAL_LENS_STATE_DOMAIN + b"\0")
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise HistoricalInputLensError(f"lens state value {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(dtype).to_bytes(8, "big"))
        digest.update(dtype)
        digest.update(len(tensor.shape).to_bytes(8, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))
    return digest.hexdigest()


def _checked_lens_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or set(state) != {"W", "b", "s"}:
        raise HistoricalInputLensError("historical lens state must contain exactly W, b, and s")
    W = state["W"]
    b = state["b"]
    s = state["s"]
    if not all(isinstance(value, torch.Tensor) for value in (W, b, s)):
        raise HistoricalInputLensError("historical lens state values must be tensors")
    if tuple(W.shape) != (HISTORICAL_HIDDEN_SIZE, HISTORICAL_HIDDEN_SIZE):
        raise HistoricalInputLensError("historical W geometry changed")
    if tuple(b.shape) != (HISTORICAL_HIDDEN_SIZE,):
        raise HistoricalInputLensError("historical b geometry changed")
    if s.ndim != 0:
        raise HistoricalInputLensError("historical s must be scalar")
    if any(value.dtype != torch.float32 for value in (W, b, s)):
        raise HistoricalInputLensError("historical lens state must remain float32")
    if not all(torch.isfinite(value).all().item() for value in (W, b, s)):
        raise HistoricalInputLensError("historical lens state contains non-finite values")
    return {
        "W": W.detach().cpu().contiguous().clone(),
        "b": b.detach().cpu().contiguous().clone(),
        "s": s.detach().cpu().contiguous().clone(),
    }


def load_historical_lens_checkpoint(path: Path, *, device: torch.device | str = "cpu") -> "HistoricalInputLensBridge":
    """Load the exact retained A1 checkpoint with a fail-closed schema check."""

    _regular_file(path, label="historical lens checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - backend-specific load errors
        raise HistoricalInputLensError(f"cannot load historical lens checkpoint: {path}") from exc
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"sd", "hidden", "corpus"}:
        raise HistoricalInputLensError("historical lens checkpoint fields changed")
    if checkpoint["hidden"] != HISTORICAL_HIDDEN_MARKER or checkpoint["corpus"] != HISTORICAL_CORPUS:
        raise HistoricalInputLensError("historical lens architecture/corpus marker changed")
    return HistoricalInputLensBridge(
        _checked_lens_state(checkpoint["sd"]),
        checkpoint_path=path,
    ).to(device).eval()


def prepare_normalized_embeddings(embedding: torch.Tensor) -> torch.Tensor:
    """Prepare raw public embeddings exactly as the historical reference does."""

    if embedding.ndim != 2 or embedding.shape[1] != HISTORICAL_HIDDEN_SIZE:
        raise HistoricalInputLensError("public embedding table must be [vocabulary,2048]")
    if not embedding.dtype.is_floating_point:
        raise HistoricalInputLensError("public embedding table must be floating point")
    normalized = F.normalize(embedding.detach().float(), dim=-1)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(normalized).all().item():
        raise HistoricalInputLensError("public embedding normalization is non-finite")
    return normalized.contiguous()


def validate_normalized_embeddings(
    embeddings: torch.Tensor,
    *,
    vocabulary_size: int | None = None,
    check_unit_norm: bool = False,
) -> None:
    """Validate a pre-normalized runtime table once at its resource boundary."""

    if embeddings.ndim != 2 or embeddings.shape[1] != HISTORICAL_HIDDEN_SIZE:
        raise HistoricalInputLensError("normalized public embeddings must be [vocabulary,2048]")
    if vocabulary_size is not None and embeddings.shape[0] != vocabulary_size:
        raise HistoricalInputLensError("normalized public embedding vocabulary changed")
    if not embeddings.dtype.is_floating_point:
        raise HistoricalInputLensError("normalized public embeddings must be floating point")
    if not torch.isfinite(embeddings).all().item():
        raise HistoricalInputLensError("normalized public embeddings are non-finite")
    if check_unit_norm:
        norms = torch.linalg.vector_norm(embeddings.float(), dim=-1)
        # F.normalize preserves an all-zero source row as zero, so permit that
        # exact reference behavior alongside ordinary unit-norm rows.
        unit = torch.ones_like(norms)
        zero = torch.zeros_like(norms)
        if not (torch.isclose(norms, unit, atol=2e-4, rtol=2e-4) | torch.isclose(norms, zero, atol=2e-6, rtol=0.0)).all().item():
            raise HistoricalInputLensError("normalized public embeddings are not unit norm")


def _runtime_activation_checks(activation: torch.Tensor) -> None:
    if activation.ndim < 1 or activation.shape[-1] != HISTORICAL_HIDDEN_SIZE:
        raise HistoricalInputLensError("activation hidden geometry changed")
    if not activation.dtype.is_floating_point:
        raise HistoricalInputLensError("activation must be floating point")
    if not torch.isfinite(activation).all().item():
        raise HistoricalInputLensError("activation contains non-finite values")


def _runtime_embedding_checks(embeddings: torch.Tensor) -> None:
    # Full-table finiteness and norm checks belong at the load boundary.  Do
    # cheap checks here so a hot direct-prediction loop does not rescan 1 GiB.
    if embeddings.ndim != 2 or embeddings.shape[1] != HISTORICAL_HIDDEN_SIZE:
        raise HistoricalInputLensError("normalized public embeddings geometry changed")
    if not embeddings.dtype.is_floating_point:
        raise HistoricalInputLensError("normalized public embeddings must be floating point")


class HistoricalInputLensBridge(nn.Module):
    """Frozen, search-free inference bridge for the retained historical A1."""

    method_id = "historical_inputlens_affine_bridge"
    spec = HistoricalInputLensSpec()

    def __init__(self, state: Mapping[str, torch.Tensor], *, checkpoint_path: Path | None = None) -> None:
        super().__init__()
        checked = _checked_lens_state(state)
        self.register_buffer("W", checked["W"])
        self.register_buffer("b", checked["b"])
        self.register_buffer("s", checked["s"])
        self.checkpoint_path = str(checkpoint_path.resolve()) if checkpoint_path is not None else None
        self.requires_grad_(False)

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
        *,
        checkpoint_path: Path | None = None,
    ) -> "HistoricalInputLensBridge":
        """Construct a frozen bridge from already validated state, without fitting."""

        return cls(state, checkpoint_path=checkpoint_path).eval()

    @property
    def lens_state_sha256(self) -> str:
        return state_sha256({"W": self.W, "b": self.b, "s": self.s})

    @property
    def logit_scale_value(self) -> float:
        return float(self.s.float().exp().item())

    def projected(self, activation: torch.Tensor) -> torch.Tensor:
        """Apply the historical hidden-space affine map in float32."""

        _runtime_activation_checks(activation)
        value = activation.float()
        return value @ self.W.float().T + self.b.float()

    def forward(self, activation: torch.Tensor, normalized_embeddings: torch.Tensor) -> torch.Tensor:
        """Return historical A1 logits; this path performs no candidate search."""

        _runtime_embedding_checks(normalized_embeddings)
        projected = F.normalize(self.projected(activation), dim=-1)
        logits = projected.to(normalized_embeddings.dtype) @ normalized_embeddings.T
        output = logits.float() * self.s.float().exp()
        if not torch.isfinite(output).all().item():
            raise HistoricalInputLensError("historical bridge logits are non-finite")
        return output

    @torch.inference_mode()
    def predict(
        self,
        activation: torch.Tensor,
        normalized_embeddings: torch.Tensor,
        *,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Emit one direct token ID per activation row, with no A2 fallback."""

        _runtime_activation_checks(activation)
        _runtime_embedding_checks(normalized_embeddings)
        if batch_size <= 0:
            raise HistoricalInputLensError("prediction batch size must be positive")
        flat = activation.reshape(-1, HISTORICAL_HIDDEN_SIZE)
        predictions: list[torch.Tensor] = []
        for start in range(0, int(flat.shape[0]), batch_size):
            logits = self(flat[start : start + batch_size], normalized_embeddings)
            predictions.append(logits.argmax(dim=-1).to(device="cpu", dtype=torch.int32))
        if not predictions:
            raise HistoricalInputLensError("prediction received no activation rows")
        return torch.cat(predictions, dim=0).reshape(activation.shape[:-1]).contiguous()

    @torch.inference_mode()
    def topk(
        self,
        activation: torch.Tensor,
        normalized_embeddings: torch.Tensor,
        *,
        k: int = 512,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Return diagnostic top-k ranks; callers must not treat this as A2."""

        _runtime_activation_checks(activation)
        _runtime_embedding_checks(normalized_embeddings)
        if k <= 0 or k > normalized_embeddings.shape[0]:
            raise HistoricalInputLensError("top-k must be within the public vocabulary")
        if batch_size <= 0:
            raise HistoricalInputLensError("top-k batch size must be positive")
        flat = activation.reshape(-1, HISTORICAL_HIDDEN_SIZE)
        values: list[torch.Tensor] = []
        for start in range(0, int(flat.shape[0]), batch_size):
            logits = self(flat[start : start + batch_size], normalized_embeddings)
            values.append(torch.topk(logits, k=k, dim=-1, sorted=True).indices.cpu().to(torch.int32))
        if not values:
            raise HistoricalInputLensError("top-k received no activation rows")
        return torch.cat(values, dim=0).reshape(*activation.shape[:-1], k).contiguous()


def equivalence_metrics(
    actual_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    actual_topk: torch.Tensor | None = None,
    reference_topk: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize logit and rank agreement without consulting token truth."""

    if actual_logits.shape != reference_logits.shape:
        raise HistoricalInputLensError("bridge/reference logit geometry differs")
    actual = actual_logits.float()
    reference = reference_logits.float()
    if not torch.isfinite(actual).all().item() or not torch.isfinite(reference).all().item():
        raise HistoricalInputLensError("bridge/reference logits are non-finite")
    error = actual - reference
    denom = torch.linalg.vector_norm(reference.reshape(-1)).clamp_min(1e-12)
    result: dict[str, Any] = {
        "elements": int(actual.numel()),
        "max_abs": float(error.abs().max().item()),
        "mean_abs": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "relative_l2": float((torch.linalg.vector_norm(error.reshape(-1)) / denom).item()),
        "exact_equal": bool(torch.equal(actual_logits, reference_logits)),
        "allclose_atol_1e-6_rtol_1e-6": bool(torch.allclose(actual, reference, atol=1e-6, rtol=1e-6)),
    }
    if actual_topk is not None or reference_topk is not None:
        if actual_topk is None or reference_topk is None or actual_topk.shape != reference_topk.shape:
            raise HistoricalInputLensError("bridge/reference rank geometry differs")
        rank_equal = actual_topk.eq(reference_topk)
        result["topk_shape"] = list(actual_topk.shape)
        result["topk_position_mismatches"] = int((~rank_equal).sum().item())
        result["topk_rows_with_mismatch"] = int((~rank_equal.all(dim=-1)).sum().item())
        result["top1_mismatches"] = int((~rank_equal[..., 0]).sum().item())
    return result


def binding_metadata(
    bridge: HistoricalInputLensBridge,
    *,
    checkpoint_path: Path,
    normalized_embedding_path: Path | None = None,
) -> dict[str, Any]:
    """Describe the fixed method state for an evidence record."""

    checkpoint = _regular_file(checkpoint_path, label="historical lens checkpoint")
    result: dict[str, Any] = {
        "schema": HISTORICAL_INPUTLENS_SCHEMA,
        "method_id": bridge.method_id,
        "spec": json.loads(json.dumps(bridge.spec.__dict__, sort_keys=True)),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "state_sha256": bridge.lens_state_sha256,
        },
        "logit_scale_value": bridge.logit_scale_value,
        "preparation": "load frozen checkpoint and public normalized embedding table; no optimization",
    }
    if normalized_embedding_path is not None:
        table = _regular_file(normalized_embedding_path, label="public embedding table")
        result["normalized_embedding_table"] = {
            "path": str(table),
            "sha256": file_sha256(table),
            "bytes": table.stat().st_size,
        }
    return result

