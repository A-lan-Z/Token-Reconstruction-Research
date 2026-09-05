"""Training-free inversion of a public causal transformer prefix.

The implementation in this module uses only the public prefix parameters and
an observed boundary activation.  A Llama decoder block has the pre-norm
residual form

    a = x + Attention(RMSNorm(x))
    y = a + MLP(RMSNorm(a))

For an observed ``y`` we invert the MLP and attention residuals in reverse
order with damped fixed-point iterations.  The initial iterate is the observed
block output itself; it is never a source-token embedding or a fitted inverse.
The recovered input is subsequently projected to the public input vocabulary.

This is deliberately separate from :mod:`token_reconstruction.inverse`, whose
``ResidualAffineInverse`` is a fitted A1 component.  No parameters in this
module are trainable and no candidate is simulated through the public prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F


class CheckpointInverseError(RuntimeError):
    """Raised when a public-prefix inversion contract is violated."""


@dataclass(frozen=True)
class FixedPointStep:
    """Public convergence diagnostic for one damped fixed-point update."""

    iteration: int
    relative_residual: float
    relative_update: float
    finite: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "iteration": self.iteration,
            "relative_residual": self.relative_residual,
            "relative_update": self.relative_update,
            "finite": self.finite,
        }


@dataclass(frozen=True)
class ResidualBranchStats:
    """Convergence trace for one MLP or attention residual branch."""

    branch: str
    layer_index: int
    steps: tuple[FixedPointStep, ...]

    @property
    def final_relative_residual(self) -> float:
        return self.steps[-1].relative_residual if self.steps else math.inf

    @property
    def finite(self) -> bool:
        return bool(self.steps) and all(step.finite for step in self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "layer_index": self.layer_index,
            "steps": [step.as_dict() for step in self.steps],
            "final_relative_residual": self.final_relative_residual,
            "finite": self.finite,
        }


@dataclass(frozen=True)
class CheckpointInverseResult:
    """Recovered public embedding sequence and all public diagnostics."""

    embedding_estimate: torch.Tensor
    branch_stats_reverse_order: tuple[ResidualBranchStats, ...]
    iterations: int
    damping: float

    @property
    def all_finite(self) -> bool:
        return bool(torch.isfinite(self.embedding_estimate).all().item()) and all(
            item.finite for item in self.branch_stats_reverse_order
        )


def _relative_l2(error: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(error.float().reshape(-1))
    denominator = torch.linalg.vector_norm(reference.float().reshape(-1)).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration as exc:
        raise CheckpointInverseError("public prefix layer has no parameters") from exc


def _causal_mask(hidden: torch.Tensor) -> torch.Tensor:
    """Build the full causal mask used by the public full-sequence forward."""

    if hidden.ndim != 3 or hidden.shape[1] <= 0:
        raise CheckpointInverseError("hidden sequence must be nonempty [batch,time,width]")
    tokens = int(hidden.shape[1])
    minimum = torch.finfo(hidden.dtype).min
    mask = torch.full(
        (tokens, tokens), minimum, dtype=hidden.dtype, device=hidden.device
    )
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, tokens, tokens).expand(hidden.shape[0], 1, tokens, tokens)


def _position_inputs(
    precut: torch.nn.Module, hidden: torch.Tensor
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    if not hasattr(precut, "rotary_emb"):
        raise CheckpointInverseError("public prefix does not expose rotary embeddings")
    batch, tokens, _ = hidden.shape
    position_ids = torch.arange(
        tokens, dtype=torch.long, device=hidden.device
    ).view(1, -1).expand(batch, -1)
    position_embeddings = precut.rotary_emb(hidden, position_ids)
    return position_ids, position_embeddings, _causal_mask(hidden)


def _hidden(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        if not output:
            raise CheckpointInverseError("public layer returned an empty tuple")
        return output[0]
    return output


def _normalization(layer: torch.nn.Module, name: str, value: torch.Tensor) -> torch.Tensor:
    normalizer = getattr(layer, name, None)
    if not callable(normalizer):
        raise CheckpointInverseError(f"public layer does not expose {name}")
    return normalizer(value)


def _mlp_branch(layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    dtype = _module_dtype(layer)
    value = hidden.to(dtype=dtype)
    mlp = getattr(layer, "mlp", None)
    if not callable(mlp):
        raise CheckpointInverseError("public layer does not expose an MLP")
    return _hidden(mlp(_normalization(layer, "post_attention_layernorm", value))).float()


def _attention_branch(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    *,
    position_embeddings: Any,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    dtype = _module_dtype(layer)
    value = hidden.to(dtype=dtype)
    attention = getattr(layer, "self_attn", None)
    if not callable(attention):
        raise CheckpointInverseError("public layer does not expose self attention")
    kwargs: dict[str, Any] = {
        "hidden_states": _normalization(layer, "input_layernorm", value),
        "position_embeddings": position_embeddings,
        "attention_mask": attention_mask,
    }
    # Transformers has used both cache keyword spellings.  This branch is a
    # full-sequence evaluation and must explicitly provide no cache when the
    # installed attention implementation accepts one.
    parameters = inspect.signature(attention.forward).parameters
    if "past_key_values" in parameters:
        kwargs["past_key_values"] = None
    elif "past_key_value" in parameters:
        kwargs["past_key_value"] = None
    return _hidden(attention(**kwargs)).float()


def _fixed_point_inverse(
    target: torch.Tensor,
    branch: Callable[[torch.Tensor], torch.Tensor],
    *,
    branch_name: str,
    layer_index: int,
    iterations: int,
    damping: float,
) -> tuple[torch.Tensor, ResidualBranchStats]:
    """Invert ``target = estimate + branch(estimate)`` by damped Jacobi steps."""

    estimate = target.float().clone()
    traces: list[FixedPointStep] = []
    for iteration in range(1, iterations + 1):
        residual_branch = branch(estimate)
        proposal = target.float() - residual_branch
        updated = estimate.lerp(proposal, damping)
        residual = updated + branch(updated) - target.float()
        finite = bool(torch.isfinite(updated).all().item()) and bool(
            torch.isfinite(residual).all().item()
        )
        traces.append(
            FixedPointStep(
                iteration=iteration,
                relative_residual=_relative_l2(residual, target),
                relative_update=_relative_l2(updated - estimate, estimate),
                finite=finite,
            )
        )
        estimate = updated
        if not finite:
            break
    return estimate, ResidualBranchStats(
        branch=branch_name,
        layer_index=layer_index,
        steps=tuple(traces),
    )


@torch.inference_mode()
def invert_public_prefix(
    precut: torch.nn.Module,
    observed_hidden: torch.Tensor,
    *,
    iterations: int,
    damping: float,
) -> CheckpointInverseResult:
    """Invert every public prefix layer in reverse order.

    ``observed_hidden`` is the activation at the declared public-prefix
    boundary.  It must be an unpadded full causal sequence.  The caller may
    process each valid prefix separately when records are right padded.  The
    returned estimate is float32 and contains no fitted state.
    """

    if observed_hidden.ndim != 3 or observed_hidden.shape[1] <= 0:
        raise CheckpointInverseError(
            "observed hidden state must be nonempty [batch,time,width]"
        )
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise CheckpointInverseError("iterations must be a positive integer")
    if not math.isfinite(float(damping)) or not 0.0 < damping <= 1.0:
        raise CheckpointInverseError("damping must lie in (0,1]")
    layers = list(getattr(precut, "layers", ()))
    if not layers:
        raise CheckpointInverseError("public prefix has no decoder layers")

    module_dtype = _module_dtype(layers[0])
    positional_hidden = observed_hidden.to(dtype=module_dtype)
    _, position_embeddings, attention_mask = _position_inputs(precut, positional_hidden)
    hidden = observed_hidden.float()
    stats: list[ResidualBranchStats] = []

    for layer_index in reversed(range(len(layers))):
        layer = layers[layer_index]
        after_attention, mlp_stats = _fixed_point_inverse(
            hidden,
            lambda value, current=layer: _mlp_branch(current, value),
            branch_name="mlp",
            layer_index=layer_index,
            iterations=iterations,
            damping=damping,
        )
        layer_input, attention_stats = _fixed_point_inverse(
            after_attention,
            lambda value, current=layer: _attention_branch(
                current,
                value,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            ),
            branch_name="attention",
            layer_index=layer_index,
            iterations=iterations,
            damping=damping,
        )
        stats.extend((mlp_stats, attention_stats))
        hidden = layer_input
        if not bool(torch.isfinite(hidden).all().item()):
            break

    return CheckpointInverseResult(
        embedding_estimate=hidden,
        branch_stats_reverse_order=tuple(stats),
        iterations=iterations,
        damping=float(damping),
    )


@torch.inference_mode()
def forward_public_embeddings(
    precut: torch.nn.Module, embedding_hidden: torch.Tensor
) -> torch.Tensor:
    """Run the public prefix from arbitrary input-embedding-space vectors."""

    if embedding_hidden.ndim != 3 or embedding_hidden.shape[1] <= 0:
        raise CheckpointInverseError(
            "embedding hidden state must be nonempty [batch,time,width]"
        )
    layers: Sequence[torch.nn.Module] = list(getattr(precut, "layers", ()))
    if not layers:
        raise CheckpointInverseError("public prefix has no decoder layers")
    dtype = _module_dtype(layers[0])
    hidden = embedding_hidden.to(dtype=dtype)
    position_ids, position_embeddings, attention_mask = _position_inputs(precut, hidden)
    for layer in layers:
        output = layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden = _hidden(output)
    return hidden


def _stable_distance_order(
    ids: torch.Tensor, distances: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort each row by distance then token ID for deterministic projection."""

    ordered_ids = torch.empty_like(ids)
    ordered_distances = torch.empty_like(distances)
    for row in range(ids.shape[0]):
        order = sorted(
            range(ids.shape[1]),
            key=lambda column: (float(distances[row, column]), int(ids[row, column])),
        )
        index = torch.tensor(order, dtype=torch.long, device=ids.device)
        ordered_ids[row] = ids[row].index_select(0, index)
        ordered_distances[row] = distances[row].index_select(0, index)
    return ordered_ids, ordered_distances


@torch.inference_mode()
def nearest_public_embeddings(
    query: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    top_k: int,
    vocab_chunk_size: int = 8192,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact chunked Euclidean or cosine-nearest public token rows.

    The vocabulary is partitioned only for memory; the result is equivalent to
    a dense scan.  ``normalize=False`` is the primary structural projection,
    while ``normalize=True`` is an explicitly labelled diagnostic variant.
    Returned distances are squared Euclidean distances for the former and
    ``1 - cosine`` for the latter.
    """

    if query.ndim < 2 or embedding_weight.ndim != 2:
        raise CheckpointInverseError("projection tensors have invalid rank")
    if query.shape[-1] != embedding_weight.shape[-1]:
        raise CheckpointInverseError("projection widths differ")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise CheckpointInverseError("top_k must be an integer")
    if not 0 < top_k <= embedding_weight.shape[0]:
        raise CheckpointInverseError("top_k is outside the vocabulary")
    if not isinstance(vocab_chunk_size, int) or vocab_chunk_size <= 0:
        raise CheckpointInverseError("vocab_chunk_size must be positive")
    if not torch.isfinite(query.float()).all().item() or not torch.isfinite(
        embedding_weight.float()
    ).all().item():
        raise CheckpointInverseError("projection tensors contain non-finite values")

    original_shape = query.shape[:-1]
    flat_query = query.reshape(-1, query.shape[-1]).float()
    vocabulary = int(embedding_weight.shape[0])
    if normalize:
        flat_query = F.normalize(flat_query, dim=-1)
    query_norm = flat_query.square().sum(dim=1, keepdim=True)
    best_distances: torch.Tensor | None = None
    best_ids: torch.Tensor | None = None

    for start in range(0, vocabulary, vocab_chunk_size):
        stop = min(vocabulary, start + vocab_chunk_size)
        chunk = embedding_weight[start:stop].float()
        if normalize:
            chunk = F.normalize(chunk, dim=-1)
            distances = 1.0 - flat_query.matmul(chunk.transpose(0, 1))
        else:
            distances = query_norm + chunk.square().sum(dim=1).view(1, -1)
            distances = distances - 2.0 * flat_query.matmul(chunk.transpose(0, 1))
        local_k = min(top_k, stop - start)
        local_distances, local_ids = torch.topk(
            distances, k=local_k, dim=1, largest=False, sorted=False
        )
        local_ids = local_ids.add(start)
        if best_distances is None:
            best_distances, best_ids = local_distances, local_ids
            continue
        merged_distances = torch.cat((best_distances, local_distances), dim=1)
        merged_ids = torch.cat((best_ids, local_ids), dim=1)
        keep = min(top_k, merged_distances.shape[1])
        best_distances, ordering = torch.topk(
            merged_distances, k=keep, dim=1, largest=False, sorted=False
        )
        best_ids = merged_ids.gather(1, ordering)

    if best_distances is None or best_ids is None:
        raise CheckpointInverseError("empty public vocabulary")
    best_ids, best_distances = _stable_distance_order(best_ids, best_distances)
    return (
        best_ids.reshape(*original_shape, top_k),
        best_distances.reshape(*original_shape, top_k),
    )


def clamp_known_bos(
    embedding_estimate: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    bos_token_id: int = 128000,
) -> torch.Tensor:
    """Replace only the declared BOS position with its public embedding."""

    if embedding_estimate.ndim != 3 or embedding_estimate.shape[1] <= 0:
        raise CheckpointInverseError("embedding estimate must be [batch,time,width]")
    if not 0 <= bos_token_id < embedding_weight.shape[0]:
        raise CheckpointInverseError("BOS token is outside the public vocabulary")
    result = embedding_estimate.clone()
    result[:, 0] = embedding_weight[bos_token_id].to(
        device=result.device, dtype=result.dtype
    )
    return result


__all__ = [
    "CheckpointInverseError",
    "CheckpointInverseResult",
    "FixedPointStep",
    "ResidualBranchStats",
    "clamp_known_bos",
    "forward_public_embeddings",
    "invert_public_prefix",
    "nearest_public_embeddings",
]
