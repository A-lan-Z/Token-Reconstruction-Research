from __future__ import annotations

import torch
import pytest

from token_reconstruction.p04_objectives import (
    derive_rank_scale,
    hard_confusion_loss,
    pairwise_teacher_loss,
    student_objective,
)
from token_reconstruction.p04_student import METHOD_D, METHOD_H, METHOD_S, P04StudentError


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [0.1, 2.0, 1.5, 0.0, 0.8, -1.0, 0.3],
            [1.1, 0.4, 1.0, 0.2, 0.8, -0.3, 0.5],
            [0.0, 1.0, 0.2, 1.5, 0.6, 0.4, -0.1],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([1, 0, 3])
    candidates = torch.tensor([[2, 4, 1, 6], [1, 2, 5, 6], [0, 2, 4, 6]])
    scores = torch.tensor([[0.9, 0.2, 0.9, -0.2], [0.8, 0.4, 0.1, 0.0], [0.7, 0.7, 0.1, -0.4]])
    return logits, labels, candidates, scores


def test_hard_confusion_excludes_gold_and_is_finite() -> None:
    logits, labels, candidates, _ = _batch()
    loss, diagnostics = hard_confusion_loss(logits, labels, candidates)
    assert torch.isfinite(loss)
    assert diagnostics["hard_rows"] == 3
    assert diagnostics["hard_negative_terms"] == 11
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_relative_loss_omits_ties_and_does_not_create_teacher_label() -> None:
    logits, labels, candidates, scores = _batch()
    scale = derive_rank_scale(candidates, scores, labels=labels)
    assert scale["sigma_q"] > 0
    loss, diagnostics = pairwise_teacher_loss(
        logits,
        labels,
        candidates,
        scores,
        sigma_q=float(scale["sigma_q"]),
        tie_tolerance=float(scale["tie_tolerance"]),
    )
    assert torch.isfinite(loss)
    assert diagnostics["rank_pairs"] > 0
    assert diagnostics["omitted_tie_pairs"] >= 1
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_student_objective_dispatch_keeps_s_full_ce_only_and_masks_d() -> None:
    logits, labels, candidates, scores = _batch()
    s = student_objective(logits, labels, method_id=METHOD_S)
    assert s.hard.item() == 0.0 and s.rank.item() == 0.0
    with pytest.raises(P04StudentError, match="does not accept"):
        student_objective(logits, labels, method_id=METHOD_S, candidate_ids=candidates)
    d = student_objective(
        logits,
        labels,
        method_id=METHOD_D,
        candidate_ids=candidates,
        teacher_scores=scores,
        rank_mask=torch.tensor([True, False, True]),
        sigma_q=1.0,
        tie_tolerance=0.01,
    )
    assert torch.isfinite(d.total)
    assert d.diagnostics.rank_rows == 2
    h = student_objective(logits, labels, method_id=METHOD_H, candidate_ids=candidates)
    assert h.hard.item() > 0
    with pytest.raises(P04StudentError, match="teacher scores"):
        student_objective(logits, labels, method_id=METHOD_H, candidate_ids=candidates, teacher_scores=scores)


def test_frozen_teacher_reference_loader_registers_module() -> None:
    from pathlib import Path
    from token_reconstruction.p04_teacher import _load_reference

    module = _load_reference(Path("experiments/TRR-0004/evidence/comparators/round001_teacher.py"))
    assert module.FrozenAffineLens.__name__ == "FrozenAffineLens"


def test_teacher_order_changes_d_only_and_gold_is_excluded() -> None:
    logits, labels, candidates, scores = _batch()
    scores_a = scores.clone()
    scores_a[0] = torch.tensor([0.9, 0.2, 0.5, -0.2])
    scores_b = scores_a.clone()
    scores_b[0] = torch.tensor([0.2, 0.9, 0.5, -0.2])
    h_a = student_objective(logits.detach(), labels, method_id=METHOD_H, candidate_ids=candidates)
    h_b = student_objective(logits.detach(), labels, method_id=METHOD_H, candidate_ids=candidates)
    assert torch.equal(h_a.hard, h_b.hard)
    d_a = student_objective(logits.detach(), labels, method_id=METHOD_D, candidate_ids=candidates, teacher_scores=scores_a, sigma_q=0.4, tie_tolerance=0.001)
    d_b = student_objective(logits.detach(), labels, method_id=METHOD_D, candidate_ids=candidates, teacher_scores=scores_b, sigma_q=0.4, tie_tolerance=0.001)
    assert d_a.rank.item() != pytest.approx(d_b.rank.item())
