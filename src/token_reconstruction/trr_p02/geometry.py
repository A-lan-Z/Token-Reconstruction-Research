"""Deterministic tensor summaries for the TRR-P02 public geometry diagnosis.

This module does not load a model, accept source truth, or train a decoder.  It
only consumes activation arrays that the P02 runner obtained from explicitly
known public token IDs.  Ranking functions are chunked so a single projected
prototype block is materialized at a time; a full per-context table is never
required.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


class GeometryDiagnosticError(RuntimeError):
    """Raised when a public geometry diagnostic violates its tensor contract."""


@dataclass(frozen=True)
class ContextSpec:
    """One explicitly known public prefix used for a teacher-prefix control."""

    name: str
    token_ids: tuple[int, ...]

    def validate(self, *, bos_token_id: int, vocab_size: int) -> None:
        if not self.name or not isinstance(self.name, str):
            raise GeometryDiagnosticError("context name must be a non-empty string")
        if not self.token_ids or self.token_ids[0] != int(bos_token_id):
            raise GeometryDiagnosticError("every context must begin with the declared BOS")
        if any(
            isinstance(token, bool) or int(token) < 0 or int(token) >= int(vocab_size)
            for token in self.token_ids
        ):
            raise GeometryDiagnosticError("context token ID is outside the public vocabulary")


def _finite_matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise GeometryDiagnosticError(f"{name} must be a rank-2 tensor")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise GeometryDiagnosticError(f"{name} must be non-empty")
    if not value.dtype.is_floating_point or not torch.isfinite(value).all().item():
        raise GeometryDiagnosticError(f"{name} must be finite floating-point data")
    return value.float()


def _finite_panel(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise GeometryDiagnosticError(f"{name} must be a rank-3 tensor")
    if min(value.shape) <= 0:
        raise GeometryDiagnosticError(f"{name} must be non-empty")
    if not value.dtype.is_floating_point or not torch.isfinite(value).all().item():
        raise GeometryDiagnosticError(f"{name} must be finite floating-point data")
    return value.float()


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float().reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all().item():
        raise GeometryDiagnosticError("summary values must be finite and non-empty")
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p10": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "max": float(values.max().item()),
    }


def _pairwise_cosine(values: torch.Tensor) -> torch.Tensor:
    values = _finite_matrix(values, name="pairwise values")
    normalized = F.normalize(values, dim=-1, eps=1e-12)
    return normalized @ normalized.transpose(0, 1)


def summarize_offsets(offsets: torch.Tensor) -> dict[str, object]:
    """Summarize shared-offset residuals for ``[contexts,tokens,hidden]``.

    The context mean is a descriptive teacher-prefix statistic.  It is never a
    reconstruction decision and callers should label any corrected ranking as
    an oracle diagnostic.
    """

    panel = _finite_panel(offsets, name="offset panel")
    contexts, tokens, _ = map(int, panel.shape)
    means = panel.mean(dim=1)
    residual = panel - means[:, None, :]
    delta_norm = torch.linalg.vector_norm(panel, dim=-1)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    ratio = residual_norm / delta_norm.clamp_min(1e-12)
    context_rows: list[dict[str, object]] = []
    for index in range(contexts):
        token_cos = _pairwise_cosine(panel[index])
        upper = token_cos[torch.triu(torch.ones_like(token_cos, dtype=torch.bool), diagonal=1)]
        context_rows.append(
            {
                "context_index": index,
                "mean_offset_norm": float(torch.linalg.vector_norm(means[index]).item()),
                "delta_norm": _summary(delta_norm[index]),
                "residual_norm": _summary(residual_norm[index]),
                "residual_to_delta_ratio": _summary(ratio[index]),
                "pairwise_offset_cosine": _summary(upper) if upper.numel() else None,
            }
        )
    return {
        "geometry": {"contexts": contexts, "tokens": tokens, "hidden": int(panel.shape[2])},
        "context_means": means,
        "residuals": residual,
        "context_rows": context_rows,
        "global_residual_to_delta_ratio": _summary(ratio),
    }


def pairwise_token_deformation(
    activations: torch.Tensor,
    *,
    token_ids: Sequence[int],
    baseline_context_index: int = 0,
) -> dict[str, object]:
    """Measure context deformation of token-pair differences.

    For each context ``C`` and pair ``(v,w)``, this returns
    ``(z(C,v)-z(C,w))-(z(C0,v)-z(C0,w))`` and compares it to the baseline pair
    difference.  No ranking or source labels are inferred here; token IDs are
    the declared public teacher-prefix controls.
    """

    panel = _finite_panel(activations, name="activation panel")
    contexts, tokens, hidden = map(int, panel.shape)
    ids = [int(value) for value in token_ids]
    if len(ids) != tokens or len(set(ids)) != len(ids):
        raise GeometryDiagnosticError("token IDs must match the activation panel and be unique")
    if not 0 <= int(baseline_context_index) < contexts:
        raise GeometryDiagnosticError("baseline context index is outside the panel")
    baseline = panel[int(baseline_context_index)]
    pair_rows: list[dict[str, object]] = []
    deformation_values: list[torch.Tensor] = []
    relative_values: list[torch.Tensor] = []
    for left, right in itertools.combinations(range(tokens), 2):
        base_pair = baseline[left] - baseline[right]
        base_norm = torch.linalg.vector_norm(base_pair).clamp_min(1e-12)
        for context_index in range(contexts):
            context_pair = panel[context_index, left] - panel[context_index, right]
            deformation = context_pair - base_pair
            deformation_norm = torch.linalg.vector_norm(deformation)
            relative = deformation_norm / base_norm
            deformation_values.append(deformation_norm)
            relative_values.append(relative)
            pair_rows.append(
                {
                    "context_index": context_index,
                    "token_id_v": ids[left],
                    "token_id_w": ids[right],
                    "baseline_pair_norm": float(base_norm.item()),
                    "deformation_norm": float(deformation_norm.item()),
                    "relative_deformation": float(relative.item()),
                    "pair_cosine_to_baseline": float(
                        F.cosine_similarity(context_pair.view(1, -1), base_pair.view(1, -1)).item()
                    ),
                }
            )
    return {
        "geometry": {"contexts": contexts, "tokens": tokens, "hidden": hidden},
        "baseline_context_index": int(baseline_context_index),
        "pair_count": len(pair_rows),
        "pairs": pair_rows,
        "deformation_norm": _summary(torch.stack(deformation_values)),
        "relative_deformation": _summary(torch.stack(relative_values)),
    }


def reference_corrected_query(
    observation: torch.Tensor,
    reference_output: torch.Tensor,
    reference_prototype: torch.Tensor,
    *,
    sign: int = -1,
) -> torch.Tensor:
    """Apply the P01 reference formula with an explicit sign.

    ``sign=-1`` is the published subtraction rule.  ``sign=+1`` is retained
    only as a wiring control and must never be interpreted as a candidate
    method.
    """

    observation = _finite_matrix(observation, name="observation")
    reference_output = _finite_matrix(reference_output, name="reference output")
    reference_prototype = _finite_matrix(reference_prototype, name="reference prototype")
    if observation.shape != reference_output.shape or reference_output.shape != reference_prototype.shape:
        raise GeometryDiagnosticError("reference tensors must have the same [rows,hidden] shape")
    if int(sign) not in (-1, 1):
        raise GeometryDiagnosticError("reference correction sign must be -1 or +1")
    return observation + int(sign) * (reference_output - reference_prototype)


def _stable_top2(
    scores: torch.Tensor, ids: torch.Tensor, *, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable top-two merge with score descending and ID ascending ties."""

    by_id = torch.argsort(ids, dim=1, stable=True)
    ids = ids.gather(1, by_id)
    scores = scores.gather(1, by_id)
    by_score = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :2]
    result_scores = scores.gather(1, by_score)
    result_ids = ids.gather(1, by_score)
    # Intermediate chunk merges may retain the sentinel until a later block;
    # the complete ranking scan checks that both final entries are real IDs.
    return result_scores, result_ids


def rank_metrics(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    true_ids: Sequence[int],
    *,
    metric: str = "cosine",
    query_chunk_size: int = 16,
    prototype_chunk_size: int = 8192,
    prototype_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return full-vocabulary ranking and margin metrics in bounded chunks.

    ``prototype_transform`` is useful for the frozen lens projected-prototype
    control.  It is applied independently to each prototype block, so no
    transformed full vocabulary table is materialized.  A single float32 score
    buffer is retained per query chunk so true-rank and tie counts cover every
    prototype block without a second transformed pass.
    """

    query = _finite_matrix(queries, name="ranking queries")
    proto = _finite_matrix(prototypes, name="ranking prototypes")
    rows, hidden = map(int, query.shape)
    vocab_size = int(proto.shape[0])
    ids = torch.tensor([int(value) for value in true_ids], dtype=torch.long)
    if ids.numel() != rows or ids.lt(0).any().item() or ids.ge(vocab_size).any().item():
        raise GeometryDiagnosticError("true IDs do not match ranking query geometry")
    if query.shape[1] != proto.shape[1]:
        raise GeometryDiagnosticError("ranking query/prototype hidden sizes differ")
    if metric not in {"cosine", "l2"}:
        raise GeometryDiagnosticError("ranking metric must be cosine or l2")
    if query_chunk_size <= 0 or prototype_chunk_size <= 0:
        raise GeometryDiagnosticError("ranking chunk sizes must be positive")

    query = query.float()
    if prototype_transform is not None:
        transformed_query = prototype_transform(query)
        query_for_scores = _finite_matrix(transformed_query, name="transformed ranking queries")
    else:
        query_for_scores = query
    if metric == "cosine":
        query_for_scores = F.normalize(query_for_scores, dim=1, eps=1e-12)
    true_scores = torch.full((rows,), -float("inf"), dtype=torch.float32)
    best_other = torch.full((rows,), -float("inf"), dtype=torch.float32)
    best_scores = torch.full((rows, 2), -float("inf"), dtype=torch.float32)
    best_ids = torch.full((rows, 2), vocab_size, dtype=torch.long)
    greater_count = torch.zeros((rows,), dtype=torch.long)
    equal_count = torch.zeros((rows,), dtype=torch.long)

    for q_start in range(0, rows, int(query_chunk_size)):
        q_stop = min(q_start + int(query_chunk_size), rows)
        q = query_for_scores[q_start:q_stop]
        q_sq = q.square().sum(dim=1, keepdim=True)
        local_true = ids[q_start:q_stop]
        # Keep only one bounded score buffer for this query chunk.  It avoids
        # a second projected-lens pass while making strict rank and tie counts
        # independent of the prototype block containing the true ID.  With the
        # predeclared Q<=16 and V=128256 this buffer is about 8 MiB.
        score_buffer = torch.empty((q.shape[0], vocab_size), dtype=torch.float32)
        current_scores = torch.full((q.shape[0], 2), -float("inf"), dtype=torch.float32)
        current_ids = torch.full((q.shape[0], 2), vocab_size, dtype=torch.long)
        for p_start in range(0, vocab_size, int(prototype_chunk_size)):
            p_stop = min(p_start + int(prototype_chunk_size), vocab_size)
            p = proto[p_start:p_stop].float()
            if prototype_transform is not None:
                p = prototype_transform(p)
                p = _finite_matrix(p, name="transformed ranking prototypes")
            if metric == "cosine":
                p = F.normalize(p, dim=1, eps=1e-12)
                block = q @ p.transpose(0, 1)
            else:
                block = -(q_sq + p.square().sum(dim=1).view(1, -1) - 2.0 * (q @ p.transpose(0, 1)))
            block = block.float()
            score_buffer[:, p_start:p_stop] = block
            block_ids = torch.arange(p_start, p_stop, dtype=torch.long).view(1, -1).expand(q.shape[0], -1)
            block_scores, block_top_ids = _stable_top2(block, block_ids, vocab_size=vocab_size)
            current_scores, current_ids = _stable_top2(
                torch.cat((current_scores, block_scores), dim=1),
                torch.cat((current_ids, block_top_ids), dim=1),
                vocab_size=vocab_size,
            )

        row_indices = torch.arange(q.shape[0], dtype=torch.long)
        local_true_scores = score_buffer[row_indices, local_true].float()
        local_greater = (score_buffer > local_true_scores[:, None]).sum(dim=1)
        local_equal = (score_buffer == local_true_scores[:, None]).sum(dim=1)
        # Remove the true token after gathering its score so the best-other
        # margin excludes it even when it is tied with another prototype.
        score_buffer[row_indices, local_true] = -float("inf")
        local_best_other = score_buffer.max(dim=1).values
        greater_count[q_start:q_stop] = local_greater
        equal_count[q_start:q_stop] = local_equal
        true_scores[q_start:q_stop] = local_true_scores
        best_other[q_start:q_stop] = local_best_other
        best_scores[q_start:q_stop] = current_scores
        best_ids[q_start:q_stop] = current_ids

    if (best_ids >= vocab_size).any().item():
        raise GeometryDiagnosticError("ranking scan did not retain two real vocabulary candidates")
    if not torch.isfinite(true_scores).all().item() or not torch.isfinite(best_other).all().item():
        raise GeometryDiagnosticError("ranking scan did not observe every true ID")
    # The true token itself contributes one equality.  The rank uses a strict
    # greater-than count; ties receive the conventional lower rank and remain
    # visible through equal_count.
    true_rank = greater_count + 1
    top1 = best_ids[:, 0]
    top1_score = best_scores[:, 0]
    runner_score = best_scores[:, 1]
    top1_margin = top1_score - runner_score
    true_other_margin = true_scores - best_other
    return {
        "top1_ids": top1,
        "runner_up_ids": best_ids[:, 1],
        "top1_scores": top1_score,
        "runner_up_scores": runner_score,
        "top1_runner_margin": top1_margin,
        "true_scores": true_scores,
        "best_other_scores": best_other,
        "true_other_margin": true_other_margin,
        "true_rank": true_rank,
        "true_equal_count": equal_count,
        "top1_is_true": top1.eq(ids),
        "true_ids": ids,
    }


def separation_summary(values: torch.Tensor) -> dict[str, object]:
    """Compare same-token cross-context and different-token within-context spread."""

    panel = _finite_panel(values, name="separation panel")
    contexts, tokens, _ = map(int, panel.shape)
    same_l2: list[torch.Tensor] = []
    same_cosine: list[torch.Tensor] = []
    different_l2: list[torch.Tensor] = []
    different_cosine: list[torch.Tensor] = []
    same_rows: list[dict[str, float | int]] = []
    different_rows: list[dict[str, float | int]] = []
    for left_context, right_context in itertools.combinations(range(contexts), 2):
        for token_index in range(tokens):
            left = panel[left_context, token_index]
            right = panel[right_context, token_index]
            l2 = torch.linalg.vector_norm(left - right)
            cosine_distance = 1.0 - F.cosine_similarity(left.view(1, -1), right.view(1, -1)).item()
            same_l2.append(l2)
            same_cosine.append(torch.tensor(cosine_distance))
            same_rows.append(
                {
                    "context_left": left_context,
                    "context_right": right_context,
                    "token_index": token_index,
                    "l2": float(l2.item()),
                    "cosine_distance": float(cosine_distance),
                }
            )
    for context_index in range(contexts):
        for left_token, right_token in itertools.combinations(range(tokens), 2):
            left = panel[context_index, left_token]
            right = panel[context_index, right_token]
            l2 = torch.linalg.vector_norm(left - right)
            cosine_distance = 1.0 - F.cosine_similarity(left.view(1, -1), right.view(1, -1)).item()
            different_l2.append(l2)
            different_cosine.append(torch.tensor(cosine_distance))
            different_rows.append(
                {
                    "context_index": context_index,
                    "token_index_v": left_token,
                    "token_index_w": right_token,
                    "l2": float(l2.item()),
                    "cosine_distance": float(cosine_distance),
                }
            )
    if not same_l2 or not different_l2:
        raise GeometryDiagnosticError("separation panel needs at least two contexts and two tokens")
    same_l2_tensor = torch.stack(same_l2)
    different_l2_tensor = torch.stack(different_l2)
    same_cosine_tensor = torch.stack(same_cosine).float()
    different_cosine_tensor = torch.stack(different_cosine).float()
    return {
        "geometry": {"contexts": contexts, "tokens": tokens, "hidden": int(panel.shape[2])},
        "same_token_cross_context": {
            "l2": _summary(same_l2_tensor),
            "cosine_distance": _summary(same_cosine_tensor),
            "rows": same_rows,
        },
        "different_token_within_context": {
            "l2": _summary(different_l2_tensor),
            "cosine_distance": _summary(different_cosine_tensor),
            "rows": different_rows,
        },
        "same_to_different_l2_mean_ratio": float(
            (same_l2_tensor.mean() / different_l2_tensor.mean().clamp_min(1e-12)).item()
        ),
        "same_to_different_cosine_distance_mean_ratio": float(
            (same_cosine_tensor.mean() / different_cosine_tensor.mean().clamp_min(1e-12)).item()
        ),
    }


def tensor_to_records(value: torch.Tensor) -> list[float | int | bool]:
    """Convert a small CPU tensor to JSON-safe scalar records."""

    value = value.detach().cpu()
    if value.dtype == torch.bool:
        return [bool(item) for item in value.reshape(-1).tolist()]
    if value.dtype.is_floating_point:
        return [float(item) for item in value.reshape(-1).tolist()]
    return [int(item) for item in value.reshape(-1).tolist()]


__all__ = [
    "ContextSpec",
    "GeometryDiagnosticError",
    "pairwise_token_deformation",
    "rank_metrics",
    "reference_corrected_query",
    "separation_summary",
    "summarize_offsets",
    "tensor_to_records",
]
