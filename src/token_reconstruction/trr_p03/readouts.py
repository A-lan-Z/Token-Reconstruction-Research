"""TRR-P03 raw, projected, and historical A1 readout paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .ranking import RankResult, rank_queries, score_block


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
BOS_TOKEN_ID = 128000
CUT_DEPTH = 4
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
LENS_SHA256 = "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742"
PROJECTED_SCHEMA = "token-reconstruction.trr-p03-projected-prototypes.v1"


class ReadoutError(RuntimeError):
    """Raised when a readout asset or score path is invalid."""


def _matrix(value: torch.Tensor, name: str, *, preserve_dtype: bool = False) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ReadoutError(f"{name} must be a rank-2 tensor")
    if min(value.shape) <= 0 or not value.dtype.is_floating_point:
        raise ReadoutError(f"{name} must be non-empty floating-point data")
    if not torch.isfinite(value).all().item():
        raise ReadoutError(f"{name} contains non-finite values")
    result = value.detach().cpu().contiguous()
    return result if preserve_dtype else result.float()


def _module_device(module: Any) -> torch.device:
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ReadoutError("readout module must expose at least one parameter") from exc


def _project(
    lens: torch.nn.Module, value: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    method = getattr(lens, "projected", None)
    if not callable(method):
        raise ReadoutError("frozen lens must expose projected(activation)")
    result = method(value.to(device=device))
    if not isinstance(result, torch.Tensor) or tuple(result.shape) != tuple(value.shape):
        raise ReadoutError("frozen lens projection changed tensor geometry")
    result = result.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(result).all().item():
        raise ReadoutError("frozen lens projection is non-finite")
    return result


@torch.inference_mode()
def project_prototypes(
    prototypes: torch.Tensor,
    lens: torch.nn.Module,
    *,
    prototype_chunk_size: int = 8192,
) -> torch.Tensor:
    """Construct the full projected prototype table once.

    The result is float32 and approximately 1 GiB for the pinned vocabulary.
    Construction is a separately recorded preparation phase and callers should
    reuse the returned table across matched and shifted target arms.
    """

    raw = _matrix(prototypes, "raw prototypes", preserve_dtype=True)
    if (
        not isinstance(prototype_chunk_size, int)
        or isinstance(prototype_chunk_size, bool)
        or prototype_chunk_size <= 0
    ):
        raise ReadoutError("prototype_chunk_size must be positive")
    device = _module_device(lens)
    projected = torch.empty(
        (int(raw.shape[0]), int(raw.shape[1])), dtype=torch.float32, device="cpu"
    )
    for start in range(0, int(raw.shape[0]), prototype_chunk_size):
        stop = min(start + prototype_chunk_size, int(raw.shape[0]))
        projected[start:stop] = _project(lens, raw[start:stop], device=device)
    if not torch.isfinite(projected).all().item():
        raise ReadoutError("projected prototype table is non-finite")
    return projected


def _rank_with_normalized_candidates(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    *,
    query_chunk_size: int,
    prototype_chunk_size: int,
) -> RankResult:
    """Rank against one prepared normalized candidate table.

    Normalizing the full candidate dictionary once avoids repeating a 128,256
    row normalization for every query chunk.  The returned table is owned by
    this call and callers that score another target can prepare/reuse their
    own persisted table at the phase boundary.
    """

    q = _matrix(queries, "queries")
    candidate_matrix = _matrix(candidates, "candidate prototypes")
    normalized = torch.nn.functional.normalize(candidate_matrix, dim=1, eps=1e-12)
    if not torch.isfinite(normalized).all().item():
        raise ReadoutError("normalized candidate table is non-finite")

    def score_normalized(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(left, dim=1, eps=1e-12) @ right.T

    return rank_queries(
        q,
        normalized,
        query_chunk_size=query_chunk_size,
        prototype_chunk_size=prototype_chunk_size,
        score_fn=score_normalized,
    )


def rank_raw(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    metric: str = "cosine",
    query_chunk_size: int = 256,
    prototype_chunk_size: int = 8192,
) -> RankResult:
    """Rank observed boundary activations against raw public prototypes."""

    if metric == "cosine":
        return _rank_with_normalized_candidates(
            queries,
            _matrix(prototypes, "raw prototypes", preserve_dtype=True),
            query_chunk_size=query_chunk_size,
            prototype_chunk_size=prototype_chunk_size,
        )
    return rank_queries(
        _matrix(queries, "queries"),
        _matrix(prototypes, "raw prototypes", preserve_dtype=True),
        metric=metric,
        query_chunk_size=query_chunk_size,
        prototype_chunk_size=prototype_chunk_size,
    )


def rank_projected(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    lens: torch.nn.Module,
    *,
    projected_prototypes: torch.Tensor | None = None,
    metric: str = "cosine",
    query_chunk_size: int = 256,
    prototype_chunk_size: int = 8192,
) -> RankResult:
    """Rank with the frozen affine lens applied to both sides.

    Pass a prepared projected table to reuse one construction across all target
    arms. If omitted, construction happens once for this call and is still
    bounded to one full transformed table.
    """

    q = _matrix(queries, "queries")
    raw = _matrix(prototypes, "raw prototypes", preserve_dtype=True)
    if projected_prototypes is None:
        transformed = project_prototypes(
            raw, lens, prototype_chunk_size=prototype_chunk_size
        )
    else:
        transformed = _matrix(projected_prototypes, "projected prototypes")
        if tuple(transformed.shape) != tuple(raw.shape):
            raise ReadoutError("projected prototype geometry differs from raw table")
    transformed_q = _project(lens, q, device=_module_device(lens))
    if metric == "cosine":
        return _rank_with_normalized_candidates(
            transformed_q,
            transformed,
            query_chunk_size=query_chunk_size,
            prototype_chunk_size=prototype_chunk_size,
        )
    return rank_queries(
        transformed_q,
        transformed,
        metric=metric,
        query_chunk_size=query_chunk_size,
        prototype_chunk_size=prototype_chunk_size,
    )


@dataclass(frozen=True)
class A1ReadoutResult:
    """Deterministic A1 ranking and the native affine score scale."""

    ranking: RankResult
    score_scale: float
    score_units: str
    cosine_equivalent_units: str


def _a1_scale(lens: torch.nn.Module) -> float:
    scale_parameter = getattr(lens, "s", None)
    if not isinstance(scale_parameter, torch.Tensor) or scale_parameter.numel() != 1:
        raise ReadoutError("historical lens must expose scalar s")
    scale = float(torch.exp(scale_parameter.detach().float()).item())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ReadoutError("historical lens exp(s) scale is invalid")
    return scale


def rank_a1(
    queries: torch.Tensor,
    lens: torch.nn.Module,
    normalized_embeddings: torch.Tensor,
    *,
    query_chunk_size: int = 256,
    prototype_chunk_size: int = 8192,
) -> A1ReadoutResult:
    """Run native frozen A1 with deterministic lowest-ID ties."""

    q = _matrix(queries, "A1 queries")
    embeddings = _matrix(normalized_embeddings, "normalized embeddings")
    if q.shape[1] != embeddings.shape[1]:
        raise ReadoutError("A1 query and embedding widths differ")
    scale = _a1_scale(lens)
    device = _module_device(lens)
    method = getattr(lens, "forward", None)
    if not callable(method):
        raise ReadoutError("historical lens is not callable")

    rows = int(q.shape[0])
    vocab = int(embeddings.shape[0])
    if vocab < 2:
        raise ReadoutError("A1 requires at least two vocabulary rows")
    predictions: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    runners: list[torch.Tensor] = []
    runner_scores: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    tie_counts: list[torch.Tensor] = []
    from .ranking import _merge_tie_counts, _stable_top_two

    # Native lens.forward consumes the complete normalized table. Keep one
    # float32 table for the A1 run so its exp(s) score units are unchanged.
    embedding_device = embeddings.to(device=device)
    for q_start in range(0, rows, query_chunk_size):
        left = q[q_start : q_start + query_chunk_size].to(device=device)
        logits = method(left, embedding_device).detach().float().cpu()
        if tuple(logits.shape) != (int(left.shape[0]), vocab):
            raise ReadoutError("historical lens returned invalid A1 vocabulary geometry")
        if not torch.isfinite(logits).all().item():
            raise ReadoutError("historical A1 logits are non-finite")
        q_rows = int(left.shape[0])
        best_scores = torch.full((q_rows, 2), -float("inf"))
        best_ids = torch.full((q_rows, 2), vocab, dtype=torch.long)
        best_value = torch.full((q_rows,), -float("inf"))
        best_count = torch.zeros((q_rows,), dtype=torch.long)
        for p_start in range(0, vocab, prototype_chunk_size):
            block = logits[:, p_start : p_start + prototype_chunk_size]
            ids = torch.arange(
                p_start, p_start + block.shape[1], dtype=torch.long
            ).view(1, -1).expand(q_rows, -1)
            block_scores, block_ids = _stable_top_two(
                torch.full((q_rows, 2), -float("inf")),
                torch.full((q_rows, 2), vocab, dtype=torch.long),
                block,
                ids,
            )
            best_scores, best_ids = _stable_top_two(
                best_scores, best_ids, block_scores, block_ids
            )
            best_value, best_count = _merge_tie_counts(best_value, best_count, block)
        predictions.append(best_ids[:, 0])
        scores.append(best_scores[:, 0])
        runners.append(best_ids[:, 1])
        runner_scores.append(best_scores[:, 1])
        margins.append(best_scores[:, 0] - best_scores[:, 1])
        tie_counts.append(best_count)
    ranking = RankResult(
        top1_ids=torch.cat(predictions),
        top1_scores=torch.cat(scores),
        runner_up_ids=torch.cat(runners),
        runner_up_scores=torch.cat(runner_scores),
        margins=torch.cat(margins),
        top1_tie_count=torch.cat(tie_counts),
    )
    ranking.validate(query_count=rows, vocab_size=vocab)
    return A1ReadoutResult(
        ranking=ranking,
        score_scale=scale,
        score_units="native_lens_exp_s_cosine",
        cosine_equivalent_units="cosine",
    )


@dataclass(frozen=True)
class ProjectedReadout:
    """Explicit fitted-origin projected prototype readout descriptor."""

    lens: torch.nn.Module
    lens_artifact_sha256: str
    score_metric: str = "cosine"

    def rank(
        self,
        queries: torch.Tensor,
        prototypes: torch.Tensor,
        *,
        projected_prototypes: torch.Tensor | None = None,
        query_chunk_size: int = 256,
        prototype_chunk_size: int = 8192,
    ) -> RankResult:
        if self.lens_artifact_sha256 != LENS_SHA256:
            raise ReadoutError("projected readout lens identity changed")
        return rank_projected(
            queries,
            prototypes,
            self.lens,
            projected_prototypes=projected_prototypes,
            metric=self.score_metric,
            query_chunk_size=query_chunk_size,
            prototype_chunk_size=prototype_chunk_size,
        )


__all__ = [
    "A1ReadoutResult",
    "BOS_TOKEN_ID",
    "CUT_DEPTH",
    "HIDDEN_SIZE",
    "LENS_SHA256",
    "MODEL_ID",
    "MODEL_REVISION",
    "PROJECTED_SCHEMA",
    "ProjectedReadout",
    "ReadoutError",
    "VOCAB_SIZE",
    "project_prototypes",
    "rank_a1",
    "rank_projected",
    "rank_raw",
]
