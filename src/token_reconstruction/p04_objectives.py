"""Frozen label and relative-score objectives for TRR-P04.

The hard-confusion term and the ranking term are training-only helpers. They
never alter deployed full-vocabulary inference, and they validate candidate
arrays before using them so candidate provenance remains an explicit input to
the training receipt rather than hidden model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .p04_student import (
    ALL_METHODS,
    METHOD_D,
    METHOD_H,
    METHOD_S,
    P04StudentError,
)


DEFAULT_HARD_WEIGHT = 0.25
DEFAULT_HARD_MARGIN = 1.0
DEFAULT_RANK_WEIGHT = 0.25
DEFAULT_STUDENT_TEMPERATURE = 1.0
DEFAULT_TIE_FRACTION = 0.01
MIN_TIE_TOLERANCE = 1.0e-6
OBJECTIVE_SCHEMA = "token-reconstruction.trr-p04-objective.v1"


@dataclass(frozen=True)
class ObjectiveDiagnostics:
    """Counts retained alongside each loss for phase receipts."""

    rows: int
    hard_rows: int = 0
    hard_negative_terms: int = 0
    rank_rows: int = 0
    rank_pairs: int = 0
    omitted_tie_pairs: int = 0
    omitted_empty_pairs: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "hard_rows": self.hard_rows,
            "hard_negative_terms": self.hard_negative_terms,
            "rank_rows": self.rank_rows,
            "rank_pairs": self.rank_pairs,
            "omitted_tie_pairs": self.omitted_tie_pairs,
            "omitted_empty_pairs": self.omitted_empty_pairs,
        }


@dataclass
class ObjectiveResult:
    """Total objective and decomposed detached diagnostics."""

    total: torch.Tensor
    ce: torch.Tensor
    hard: torch.Tensor
    rank: torch.Tensor
    diagnostics: ObjectiveDiagnostics

    def scalar_dict(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "ce": float(self.ce.detach().cpu().item()),
            "hard": float(self.hard.detach().cpu().item()),
            "rank": float(self.rank.detach().cpu().item()),
            "total": float(self.total.detach().cpu().item()),
        }
        result.update(self.diagnostics.as_dict())
        return result


def _check_logits_labels(logits: torch.Tensor, labels: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[0] <= 0 or logits.shape[1] <= 0:
        raise P04StudentError("objective logits must be [rows,vocab]")
    if not logits.dtype.is_floating_point or not torch.isfinite(logits).all().item():
        raise P04StudentError("objective logits must be finite floating point")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise P04StudentError("objective labels must match logits rows")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise P04StudentError("objective labels must be integer token IDs")
    labels = labels.to(device=logits.device, dtype=torch.int64)
    if ((labels < 0) | (labels >= logits.shape[1])).any().item():
        raise P04StudentError("objective labels contain an out-of-range token ID")


def validate_candidate_arrays(
    candidate_ids: torch.Tensor,
    *,
    rows: int,
    vocab_size: int,
    teacher_scores: torch.Tensor | None = None,
) -> None:
    """Validate fixed-width, unique, full-vocabulary candidate rows."""

    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != rows or candidate_ids.shape[1] <= 0:
        raise P04StudentError("candidate IDs must be [rows,K>0]")
    if candidate_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise P04StudentError("candidate IDs must be integer token IDs")
    ids = candidate_ids.to(dtype=torch.int64)
    if ((ids < 0) | (ids >= vocab_size)).any().item():
        raise P04StudentError("candidate IDs contain an out-of-range token ID")
    if ids.shape[1] > 1 and ids.sort(dim=1).values[:, 1:].eq(ids.sort(dim=1).values[:, :-1]).any().item():
        raise P04StudentError("candidate IDs must be unique within each row")
    if teacher_scores is not None:
        if teacher_scores.shape != candidate_ids.shape:
            raise P04StudentError("teacher scores must match candidate ID geometry")
        if not teacher_scores.dtype.is_floating_point or not torch.isfinite(teacher_scores).all().item():
            raise P04StudentError("teacher scores must be finite floating point")


def hard_confusion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    margin: float = DEFAULT_HARD_MARGIN,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Label-derived hard-negative hinge over fixed candidate identities.

    The gold token is excluded from the negative set. Each row contributes
    the mean over its own available negatives, then rows are averaged so an
    accidental gold inclusion cannot change the weight of another row.
    """

    _check_logits_labels(logits, labels)
    if not torch.isfinite(torch.tensor(float(margin))) or margin < 0:
        raise P04StudentError("hard-negative margin must be finite and non-negative")
    candidate_ids = candidate_ids.to(device=logits.device, dtype=torch.int64)
    labels = labels.to(device=logits.device, dtype=torch.int64)
    validate_candidate_arrays(
        candidate_ids,
        rows=int(logits.shape[0]),
        vocab_size=int(logits.shape[1]),
    )
    gold = logits.gather(1, labels[:, None]).squeeze(1)
    candidate_logits = logits.gather(1, candidate_ids)
    negative = candidate_ids.ne(labels[:, None])
    terms = F.softplus(float(margin) + candidate_logits - gold[:, None])
    counts = negative.sum(dim=1)
    row_losses = (terms * negative).sum(dim=1) / counts.clamp_min(1)
    active_rows = counts.gt(0)
    if active_rows.any().item():
        loss = row_losses[active_rows].mean()
    else:
        loss = logits.sum() * 0.0
    diagnostics = {
        "hard_rows": int(active_rows.sum().detach().cpu().item()),
        "hard_negative_terms": int(counts.sum().detach().cpu().item()),
    }
    return loss, diagnostics


def derive_rank_scale(
    candidate_ids: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    tie_tolerance_floor: float = MIN_TIE_TOLERANCE,
) -> dict[str, float | int]:
    """Derive the single robust adjacent-gap scale used by D.

    Scores are sorted descending per row after excluding a supplied gold label.
    The median of finite nonzero adjacent gaps is frozen before D fitting. No
    score is converted into a teacher token target here.
    """

    if teacher_scores.ndim != 2:
        raise P04StudentError("teacher scores must be [rows,K]")
    rows, _ = map(int, teacher_scores.shape)
    validate_candidate_arrays(
        candidate_ids,
        rows=rows,
        vocab_size=max(int(candidate_ids.max().detach().cpu().item()) + 1, 1),
        teacher_scores=teacher_scores,
    )
    labels_cpu: list[int | None]
    if labels is None:
        labels_cpu = [None] * rows
    else:
        if labels.ndim != 1 or labels.shape[0] != rows:
            raise P04StudentError("rank-scale labels must match score rows")
        labels_cpu = [int(value) for value in labels.detach().cpu().tolist()]
    ids_cpu = candidate_ids.detach().cpu().to(torch.int64)
    scores_cpu = teacher_scores.detach().cpu().float()
    gaps: list[float] = []
    omitted_ties = 0
    for row in range(rows):
        entries = [
            (float(scores_cpu[row, col]), int(ids_cpu[row, col]), col)
            for col in range(int(scores_cpu.shape[1]))
            if labels_cpu[row] is None or int(ids_cpu[row, col]) != labels_cpu[row]
        ]
        entries.sort(key=lambda item: (-item[0], item[1], item[2]))
        for left, right in zip(entries, entries[1:]):
            gap = left[0] - right[0]
            if gap <= float(tie_tolerance_floor):
                omitted_ties += 1
            elif gap > 0:
                gaps.append(gap)
    if not gaps:
        raise P04StudentError("cannot derive ranking scale from zero nonzero score gaps")
    scale = float(torch.tensor(gaps, dtype=torch.float64).median().item())
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
        raise P04StudentError("derived ranking scale is invalid")
    tie_tolerance = max(float(tie_tolerance_floor), DEFAULT_TIE_FRACTION * scale)
    return {
        "sigma_q": scale,
        "tie_tolerance": tie_tolerance,
        "nonzero_adjacent_gaps": len(gaps),
        "scale_omitted_ties": omitted_ties,
    }


def pairwise_teacher_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_ids: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    sigma_q: float,
    tie_tolerance: float | None = None,
    student_temperature: float = DEFAULT_STUDENT_TEMPERATURE,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Relative non-gold score loss with deterministic adjacent pairs.

    Candidate rows are sorted by frozen teacher score. Equal/near-tied adjacent
    rows are omitted; token ID and original candidate order only serialize the
    sort and never create a pair for a tie. Teacher values affect pair order,
    sign, and capped weight, not a copied hard target.
    """

    _check_logits_labels(logits, labels)
    if not torch.isfinite(torch.tensor(float(sigma_q))) or sigma_q <= 0:
        raise P04StudentError("sigma_q must be finite and positive")
    if not torch.isfinite(torch.tensor(float(student_temperature))) or student_temperature <= 0:
        raise P04StudentError("student temperature must be finite and positive")
    tie_tolerance = max(MIN_TIE_TOLERANCE, DEFAULT_TIE_FRACTION * sigma_q) if tie_tolerance is None else float(tie_tolerance)
    if not torch.isfinite(torch.tensor(tie_tolerance)) or tie_tolerance < 0:
        raise P04StudentError("rank tie tolerance must be finite and non-negative")
    candidate_ids = candidate_ids.to(device=logits.device, dtype=torch.int64)
    teacher_scores = teacher_scores.to(device=logits.device, dtype=torch.float32)
    labels = labels.to(device=logits.device, dtype=torch.int64)
    validate_candidate_arrays(
        candidate_ids,
        rows=int(logits.shape[0]),
        vocab_size=int(logits.shape[1]),
        teacher_scores=teacher_scores,
    )
    candidate_logits = logits.gather(1, candidate_ids) / float(student_temperature)
    logp = F.log_softmax(candidate_logits, dim=1)
    labels_cpu = [int(value) for value in labels.detach().cpu().tolist()]
    ids_cpu = candidate_ids.detach().cpu().tolist()
    scores_cpu = teacher_scores.detach().cpu().tolist()
    pair_rows: list[int] = []
    pair_left: list[int] = []
    pair_right: list[int] = []
    pair_sign: list[float] = []
    pair_weight: list[float] = []
    omitted_ties = 0
    for row, (row_ids, row_scores) in enumerate(zip(ids_cpu, scores_cpu)):
        entries = [
            (float(score), int(token_id), col)
            for col, (token_id, score) in enumerate(zip(row_ids, row_scores))
            if int(token_id) != labels_cpu[row]
        ]
        entries.sort(key=lambda item: (-item[0], item[1], item[2]))
        if len(entries) < 2:
            continue
        for left, right in zip(entries, entries[1:]):
            delta = left[0] - right[0]
            if delta <= tie_tolerance:
                omitted_ties += 1
                continue
            pair_rows.append(row)
            pair_left.append(left[2])
            pair_right.append(right[2])
            pair_sign.append(1.0 if delta > 0 else -1.0)
            pair_weight.append(min(abs(delta) / float(sigma_q), 1.0))
    if not pair_rows:
        loss = logits.sum() * 0.0
        return loss, {
            "rank_rows": 0,
            "rank_pairs": 0,
            "omitted_tie_pairs": omitted_ties,
            "omitted_empty_pairs": int(logits.shape[0]),
        }
    row_index = torch.tensor(pair_rows, dtype=torch.int64, device=logits.device)
    left_index = torch.tensor(pair_left, dtype=torch.int64, device=logits.device)
    right_index = torch.tensor(pair_right, dtype=torch.int64, device=logits.device)
    signs = torch.tensor(pair_sign, dtype=logits.dtype, device=logits.device)
    weights = torch.tensor(pair_weight, dtype=logits.dtype, device=logits.device)
    pair_margin = logp[row_index, left_index] - logp[row_index, right_index]
    terms = F.softplus(-signs * pair_margin)
    loss = (terms * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return loss, {
        "rank_rows": len(set(pair_rows)),
        "rank_pairs": len(pair_rows),
        "omitted_tie_pairs": omitted_ties,
        "omitted_empty_pairs": 0,
    }


def student_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    method_id: str,
    candidate_ids: torch.Tensor | None = None,
    teacher_scores: torch.Tensor | None = None,
    rank_mask: torch.Tensor | None = None,
    hard_weight: float = DEFAULT_HARD_WEIGHT,
    hard_margin: float = DEFAULT_HARD_MARGIN,
    rank_weight: float = DEFAULT_RANK_WEIGHT,
    sigma_q: float | None = None,
    tie_tolerance: float | None = None,
    student_temperature: float = DEFAULT_STUDENT_TEMPERATURE,
) -> ObjectiveResult:
    """Compute one fixed P04 objective for a full-vocabulary batch.

    ``rank_mask`` identifies the subset with qualified teacher arrays; it is
    necessary because teacher evidence is intentionally limited to 384 public
    correction positions while H candidates may cover the whole schedule.
    """

    _check_logits_labels(logits, labels)
    if method_id not in (*ALL_METHODS,):
        raise P04StudentError(f"unknown objective method: {method_id}")
    if not torch.isfinite(torch.tensor(float(hard_weight))) or hard_weight < 0:
        raise P04StudentError("hard objective weight must be finite and non-negative")
    if not torch.isfinite(torch.tensor(float(rank_weight))) or rank_weight < 0:
        raise P04StudentError("rank objective weight must be finite and non-negative")
    ce = F.cross_entropy(logits, labels.to(device=logits.device, dtype=torch.int64))
    zero = logits.sum() * 0.0
    hard = zero
    rank = zero
    hard_diag = {"hard_rows": 0, "hard_negative_terms": 0}
    rank_diag = {"rank_rows": 0, "rank_pairs": 0, "omitted_tie_pairs": 0, "omitted_empty_pairs": 0}
    needs_candidates = method_id in (METHOD_H, METHOD_D)
    if not needs_candidates and candidate_ids is not None:
        raise P04StudentError(f"{method_id} does not accept candidate IDs")
    if needs_candidates and candidate_ids is None:
        raise P04StudentError(f"{method_id} requires frozen candidate IDs during training")
    if needs_candidates:
        hard, hard_diag = hard_confusion_loss(
            logits,
            labels,
            candidate_ids,
            margin=hard_margin,
        )
    if method_id == METHOD_D:
        if teacher_scores is None:
            raise P04StudentError("student_d requires frozen teacher scores during training")
        if teacher_scores.shape != candidate_ids.shape:
            raise P04StudentError("teacher scores must match candidate IDs for student_d")
        if rank_mask is None:
            rank_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        if rank_mask.shape != (logits.shape[0],) or rank_mask.dtype is not torch.bool:
            raise P04StudentError("rank mask must be boolean [rows]")
        active = rank_mask.to(device=logits.device)
        if active.any().item():
            active_ids = candidate_ids[active]
            active_scores = teacher_scores[active]
            active_labels = labels.to(device=logits.device, dtype=torch.int64)[active]
            if sigma_q is None:
                raise P04StudentError("student_d requires frozen sigma_q")
            rank, rank_diag = pairwise_teacher_loss(
                logits[active],
                active_labels,
                active_ids,
                active_scores,
                sigma_q=sigma_q,
                tie_tolerance=tie_tolerance,
                student_temperature=student_temperature,
            )
    elif teacher_scores is not None:
        raise P04StudentError("teacher scores are not allowed for affine/S/H objectives")
    total = ce + float(hard_weight) * hard + float(rank_weight) * rank
    diagnostics = ObjectiveDiagnostics(
        rows=int(logits.shape[0]),
        **hard_diag,
        **rank_diag,
    )
    return ObjectiveResult(total=total, ce=ce, hard=hard, rank=rank, diagnostics=diagnostics)


__all__ = [
    "DEFAULT_HARD_MARGIN",
    "DEFAULT_HARD_WEIGHT",
    "DEFAULT_RANK_WEIGHT",
    "DEFAULT_STUDENT_TEMPERATURE",
    "DEFAULT_TIE_FRACTION",
    "MIN_TIE_TOLERANCE",
    "OBJECTIVE_SCHEMA",
    "ObjectiveDiagnostics",
    "ObjectiveResult",
    "derive_rank_scale",
    "hard_confusion_loss",
    "pairwise_teacher_loss",
    "student_objective",
    "validate_candidate_arrays",
]
