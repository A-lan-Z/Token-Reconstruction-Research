"""Full-vocabulary decision-boundary diagnostics for TRR-P12.

The target trajectory is evaluated with a frozen decoder.  This module does
not load a decoder, observations, source text, or target weights; it consumes
already materialized score rows.  Its central invariant is that all winners
and runners-up are selected over the complete vocabulary with the deployed
``torch.argmax`` tie rule.  A score movement is retained as a diagnostic, but
only an observed argmax transition with a strict signed margin flip is called
a crossing.

For a baseline score row ``b`` and an updated score row ``u`` let ``w`` be the
baseline argmax and ``c`` be the updated argmax when it changes, otherwise the
baseline runner-up.  The transition margin is

``m_before = b[w] - b[c]`` and ``m_after = u[w] - u[c]``.

``m_before >= 0`` and ``m_after < 0`` is a strict crossing.  An argmax change
with ``m_after == 0`` is reported as a tie transition because the tie rule,
rather than a strict score reversal, selected the new ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


SCHEMA = "token-reconstruction.trr-p12-boundary-rows.v1"
VOCABULARY_SIZE = 128_256
STORED_SEQUENCE_TOKENS = 128
BOS_TOKEN_ID = 128_000


class BoundaryError(ValueError):
    """Raised when score geometry or argmax alignment is not trustworthy."""


def _scores(value: torch.Tensor, *, label: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim < 2:
        raise BoundaryError(f"{label} must have at least two dimensions")
    if not value.dtype.is_floating_point:
        raise BoundaryError(f"{label} must be floating point")
    if int(value.shape[-1]) < 2:
        raise BoundaryError(f"{label} must contain at least two vocabulary entries")
    if not bool(torch.isfinite(value).all().item()):
        raise BoundaryError(f"{label} contains non-finite scores")
    return value.detach()


def _same_prefix(left: torch.Tensor, right: torch.Tensor) -> None:
    if tuple(left.shape) != tuple(right.shape):
        raise BoundaryError(
            f"baseline and updated score geometry differs: {tuple(left.shape)} vs {tuple(right.shape)}"
        )


def _valid_mask(value: torch.Tensor | None, *, prefix: tuple[int, ...]) -> torch.Tensor:
    if value is None:
        return torch.ones(prefix, dtype=torch.bool)
    mask = torch.as_tensor(value)
    if tuple(mask.shape) != prefix:
        raise BoundaryError(f"valid mask geometry differs from score rows: {tuple(mask.shape)} vs {prefix}")
    if mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise BoundaryError("valid mask must be boolean or integer")
    return mask.to(device="cpu", dtype=torch.bool).detach()


def _flatten(value: torch.Tensor) -> torch.Tensor:
    return value.float().reshape(-1, int(value.shape[-1]))


def _argmax(value: torch.Tensor) -> torch.Tensor:
    # Keep this explicit: torch.argmax is the package's registered tie rule.
    return torch.argmax(value, dim=-1).to(dtype=torch.long)


def _gather_rows(scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return scores.gather(1, indices.reshape(-1, 1)).squeeze(1)


def _runner_up(scores: torch.Tensor, winner: torch.Tensor) -> torch.Tensor:
    masked = scores.clone()
    rows = torch.arange(int(scores.shape[0]), dtype=torch.long, device=scores.device)
    masked[rows, winner] = float("-inf")
    return _argmax(masked)


def _check_prediction_alignment(
    provided: torch.Tensor | None,
    expected: torch.Tensor,
    *,
    label: str,
    prefix: tuple[int, ...],
) -> None:
    if provided is None:
        return
    value = torch.as_tensor(provided)
    if tuple(value.shape) != prefix:
        raise BoundaryError(f"{label} geometry differs from score rows")
    value = value.to(device="cpu", dtype=torch.long).reshape(-1)
    if not torch.equal(value, expected.to(device="cpu")):
        mismatch = int(torch.nonzero(value.ne(expected.to(device="cpu")), as_tuple=False)[0].item())
        raise BoundaryError(f"{label} is not the full-vocabulary argmax at flattened row {mismatch}")


@dataclass(frozen=True)
class BoundaryRows:
    """Compact full-vocabulary boundary evidence for aligned score rows."""

    row_shape: tuple[int, ...]
    valid: torch.Tensor
    baseline_winner: torch.Tensor
    baseline_runner_up: torch.Tensor
    updated_winner: torch.Tensor
    updated_runner_up: torch.Tensor
    transition_competitor: torch.Tensor
    baseline_winner_score: torch.Tensor
    baseline_runner_up_score: torch.Tensor
    updated_winner_score: torch.Tensor
    updated_runner_up_score: torch.Tensor
    baseline_competitor_score: torch.Tensor
    updated_competitor_score: torch.Tensor
    transition_margin_before: torch.Tensor
    transition_margin_after: torch.Tensor
    signed_displacement: torch.Tensor
    runner_up_margin_before: torch.Tensor
    runner_up_margin_after: torch.Tensor
    runner_up_signed_displacement: torch.Tensor
    unsigned_pair_movement: torch.Tensor
    argmax_changed: torch.Tensor
    strict_crossing: torch.Tensor
    tie_transition: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.valid.numel())

    def _valid_values(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(-1)[self.valid.reshape(-1)]

    def summary(self) -> dict[str, Any]:
        """Return aggregate counts and signed geometry over valid rows."""

        valid_count = int(self.valid.sum().item())
        if valid_count <= 0:
            raise BoundaryError("boundary summary has no valid rows")
        def count(value: torch.Tensor) -> int:
            return int(self._valid_values(value.to(dtype=torch.bool)).sum().item())

        def mean(value: torch.Tensor) -> float:
            return float(self._valid_values(value.float()).mean().item())

        return {
            "schema": SCHEMA,
            "rows": valid_count,
            "row_shape": list(self.row_shape),
            "argmax_changed": count(self.argmax_changed),
            "strict_crossings": count(self.strict_crossing),
            "tie_transitions": count(self.tie_transition),
            "transition_change_rate": float(count(self.argmax_changed)) / valid_count,
            "strict_crossing_rate": float(count(self.strict_crossing)) / valid_count,
            "tie_transition_rate": float(count(self.tie_transition)) / valid_count,
            "transition_margin_before_mean": mean(self.transition_margin_before),
            "transition_margin_after_mean": mean(self.transition_margin_after),
            "signed_displacement_mean": mean(self.signed_displacement),
            "runner_up_margin_before_mean": mean(self.runner_up_margin_before),
            "runner_up_margin_after_mean": mean(self.runner_up_margin_after),
            "runner_up_signed_displacement_mean": mean(self.runner_up_signed_displacement),
            "unsigned_pair_movement_mean": mean(self.unsigned_pair_movement),
            "interpretation": {
                "strict_crossing": "baseline and updated full-vocabulary argmax differ and the aligned transition margin changes from nonnegative to strictly negative",
                "tie_transition": "argmax differs but the aligned updated transition margin is exactly zero under torch.argmax tie handling",
                "signed_displacement": "updated baseline-winner-vs-competitor margin minus baseline margin; negative moves toward the competitor",
                "unsigned_pair_movement": "diagnostic magnitude only; it is not evidence of a boundary crossing",
            },
        }

    def compact(self, *, include_scores: bool = True) -> dict[str, Any]:
        """Serialize the retained compact rows without returning raw logits."""

        def ids(value: torch.Tensor) -> list[int]:
            return [int(item) for item in value.reshape(-1).tolist()]

        def values(value: torch.Tensor) -> list[float]:
            return [float(item) for item in value.reshape(-1).tolist()]

        result: dict[str, Any] = {
            "schema": SCHEMA,
            "row_shape": list(self.row_shape),
            "valid": [bool(item) for item in self.valid.reshape(-1).tolist()],
            "baseline_winner": ids(self.baseline_winner),
            "baseline_runner_up": ids(self.baseline_runner_up),
            "updated_winner": ids(self.updated_winner),
            "updated_runner_up": ids(self.updated_runner_up),
            "transition_competitor": ids(self.transition_competitor),
            "argmax_changed": [bool(item) for item in self.argmax_changed.reshape(-1).tolist()],
            "strict_crossing": [bool(item) for item in self.strict_crossing.reshape(-1).tolist()],
            "tie_transition": [bool(item) for item in self.tie_transition.reshape(-1).tolist()],
            "transition_margin_before": values(self.transition_margin_before),
            "transition_margin_after": values(self.transition_margin_after),
            "signed_displacement": values(self.signed_displacement),
            "runner_up_margin_before": values(self.runner_up_margin_before),
            "runner_up_margin_after": values(self.runner_up_margin_after),
            "runner_up_signed_displacement": values(self.runner_up_signed_displacement),
            "unsigned_pair_movement": values(self.unsigned_pair_movement),
        }
        if include_scores:
            result.update(
                {
                    "baseline_winner_score": values(self.baseline_winner_score),
                    "baseline_runner_up_score": values(self.baseline_runner_up_score),
                    "updated_winner_score": values(self.updated_winner_score),
                    "updated_runner_up_score": values(self.updated_runner_up_score),
                    "baseline_competitor_score": values(self.baseline_competitor_score),
                    "updated_competitor_score": values(self.updated_competitor_score),
                }
            )
        return result


def analyze_score_pair(
    baseline_scores: torch.Tensor,
    updated_scores: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    baseline_predictions: torch.Tensor | None = None,
    updated_predictions: torch.Tensor | None = None,
) -> BoundaryRows:
    """Compute exact full-vocabulary boundary evidence for aligned score rows.

    ``baseline_scores`` and ``updated_scores`` may be ``[rows, vocab]`` or
    ``[records, positions, vocab]``.  The vocabulary dimension is always
    searched in full.  ``valid_mask`` can exclude BOS/padded rows while
    preserving the original row shape for later source pairing.
    """

    baseline = _scores(baseline_scores, label="baseline scores")
    updated = _scores(updated_scores, label="updated scores")
    _same_prefix(baseline, updated)
    row_shape = tuple(int(item) for item in baseline.shape[:-1])
    valid = _valid_mask(valid_mask, prefix=row_shape)
    base = _flatten(baseline)
    update = _flatten(updated)
    base_winner = _argmax(base)
    update_winner = _argmax(update)
    base_runner = _runner_up(base, base_winner)
    update_runner = _runner_up(update, update_winner)
    _check_prediction_alignment(
        baseline_predictions,
        base_winner.reshape(row_shape),
        label="baseline predictions",
        prefix=row_shape,
    )
    _check_prediction_alignment(
        updated_predictions,
        update_winner.reshape(row_shape),
        label="updated predictions",
        prefix=row_shape,
    )

    changed = base_winner.ne(update_winner)
    competitor = torch.where(changed, update_winner, base_runner)
    base_winner_score = _gather_rows(base, base_winner)
    base_runner_score = _gather_rows(base, base_runner)
    update_winner_score = _gather_rows(update, update_winner)
    update_runner_score = _gather_rows(update, update_runner)
    base_competitor_score = _gather_rows(base, competitor)
    update_competitor_score = _gather_rows(update, competitor)
    transition_before = base_winner_score - base_competitor_score
    transition_after = _gather_rows(update, base_winner) - update_competitor_score
    signed_displacement = transition_after - transition_before
    runner_up_before = base_winner_score - base_runner_score
    runner_up_after = _gather_rows(update, base_winner) - _gather_rows(update, base_runner)
    runner_up_displacement = runner_up_after - runner_up_before
    unsigned_movement = (torch.abs(_gather_rows(update, base_winner) - base_winner_score) + torch.abs(update_competitor_score - base_competitor_score))
    strict_crossing = changed & transition_after.lt(0.0) & transition_before.ge(0.0)
    tie_transition = changed & transition_after.eq(0.0)
    return BoundaryRows(
        row_shape=row_shape,
        valid=valid.reshape(-1),
        baseline_winner=base_winner,
        baseline_runner_up=base_runner,
        updated_winner=update_winner,
        updated_runner_up=update_runner,
        transition_competitor=competitor,
        baseline_winner_score=base_winner_score,
        baseline_runner_up_score=base_runner_score,
        updated_winner_score=update_winner_score,
        updated_runner_up_score=update_runner_score,
        baseline_competitor_score=base_competitor_score,
        updated_competitor_score=update_competitor_score,
        transition_margin_before=transition_before,
        transition_margin_after=transition_after,
        signed_displacement=signed_displacement,
        runner_up_margin_before=runner_up_before,
        runner_up_margin_after=runner_up_after,
        runner_up_signed_displacement=runner_up_displacement,
        unsigned_pair_movement=unsigned_movement,
        argmax_changed=changed,
        strict_crossing=strict_crossing,
        tie_transition=tie_transition,
    )


def top2_full_vocabulary(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return exact argmax/runner-up IDs and scores over the full vocabulary."""

    value = _scores(scores, label="scores")
    flat = _flatten(value)
    winner = _argmax(flat)
    runner = _runner_up(flat, winner)
    return {
        "winner": winner.reshape(value.shape[:-1]),
        "runner_up": runner.reshape(value.shape[:-1]),
        "winner_score": _gather_rows(flat, winner).reshape(value.shape[:-1]),
        "runner_up_score": _gather_rows(flat, runner).reshape(value.shape[:-1]),
    }


def result_from_compact(payload: Mapping[str, Any]) -> BoundaryRows:
    """Rehydrate a compact artifact for analysis without raw score access."""

    if payload.get("schema") != SCHEMA:
        raise BoundaryError("boundary compact schema is not registered")
    row_shape_value = payload.get("row_shape")
    if not isinstance(row_shape_value, Sequence) or isinstance(row_shape_value, (str, bytes)):
        raise BoundaryError("compact row_shape is malformed")
    row_shape = tuple(int(item) for item in row_shape_value)
    rows = 1
    for item in row_shape:
        if item <= 0:
            raise BoundaryError("compact row_shape must be positive")
        rows *= item

    def tensor(name: str, *, dtype: torch.dtype) -> torch.Tensor:
        value = payload.get(name)
        if not isinstance(value, list) or len(value) != rows:
            raise BoundaryError(f"compact {name} has the wrong row count")
        return torch.tensor(value, dtype=dtype)

    valid = tensor("valid", dtype=torch.bool)
    result = BoundaryRows(
        row_shape=row_shape,
        valid=valid,
        baseline_winner=tensor("baseline_winner", dtype=torch.long),
        baseline_runner_up=tensor("baseline_runner_up", dtype=torch.long),
        updated_winner=tensor("updated_winner", dtype=torch.long),
        updated_runner_up=tensor("updated_runner_up", dtype=torch.long),
        transition_competitor=tensor("transition_competitor", dtype=torch.long),
        baseline_winner_score=tensor("baseline_winner_score", dtype=torch.float32),
        baseline_runner_up_score=tensor("baseline_runner_up_score", dtype=torch.float32),
        updated_winner_score=tensor("updated_winner_score", dtype=torch.float32),
        updated_runner_up_score=tensor("updated_runner_up_score", dtype=torch.float32),
        baseline_competitor_score=tensor("baseline_competitor_score", dtype=torch.float32),
        updated_competitor_score=tensor("updated_competitor_score", dtype=torch.float32),
        transition_margin_before=tensor("transition_margin_before", dtype=torch.float32),
        transition_margin_after=tensor("transition_margin_after", dtype=torch.float32),
        signed_displacement=tensor("signed_displacement", dtype=torch.float32),
        runner_up_margin_before=tensor("runner_up_margin_before", dtype=torch.float32),
        runner_up_margin_after=tensor("runner_up_margin_after", dtype=torch.float32),
        runner_up_signed_displacement=tensor("runner_up_signed_displacement", dtype=torch.float32),
        unsigned_pair_movement=tensor("unsigned_pair_movement", dtype=torch.float32),
        argmax_changed=tensor("argmax_changed", dtype=torch.bool),
        strict_crossing=tensor("strict_crossing", dtype=torch.bool),
        tie_transition=tensor("tie_transition", dtype=torch.bool),
    )
    return result



def geometric_boundary_metrics(
    rows: BoundaryRows,
    baseline_projected_hidden: torch.Tensor,
    updated_projected_hidden: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    logit_scale: float | torch.Tensor,
    tolerance: float = 2e-3,
) -> dict[str, Any]:
    """Compute signed distances to the actual winner/competitor hyperplane.

    For normalized public-readout rows, the score boundary between IDs ``w``
    and ``c`` has normal ``E[w] - E[c]``.  The signed distance is
    ``z dot normal / ||normal||``; its sign agrees with the score margin when
    the positive logit scale is applied.  ``c`` is the exact full-vocabulary
    transition competitor retained by :func:`analyze_score_pair`, never an
    approximate nearest-neighbour ID.
    """

    base_z = torch.as_tensor(baseline_projected_hidden).detach().float()
    update_z = torch.as_tensor(updated_projected_hidden).detach().float()
    if tuple(base_z.shape) != tuple(update_z.shape) or tuple(base_z.shape[:-1]) != rows.row_shape:
        raise BoundaryError("projected hidden geometry differs from boundary rows")
    if base_z.shape[-1] <= 0 or not bool(torch.isfinite(base_z).all().item()) or not bool(torch.isfinite(update_z).all().item()):
        raise BoundaryError("projected hidden rows are non-finite or empty")
    table = torch.as_tensor(embedding_table).detach().float()
    if table.ndim != 2 or int(table.shape[1]) != int(base_z.shape[-1]) or int(table.shape[0]) <= int(rows.baseline_winner.max().item()):
        raise BoundaryError("embedding table geometry is incompatible with boundary rows")
    if not bool(torch.isfinite(table).all().item()):
        raise BoundaryError("embedding table contains non-finite values")
    scale = float(torch.as_tensor(logit_scale).item())
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0.0:
        raise BoundaryError("logit scale must be finite and positive")
    base_flat = base_z.reshape(-1, int(base_z.shape[-1]))
    update_flat = update_z.reshape(-1, int(update_z.shape[-1]))
    winner = rows.baseline_winner.reshape(-1).to(dtype=torch.long)
    competitor = rows.transition_competitor.reshape(-1).to(dtype=torch.long)
    normal = table.index_select(0, winner) - table.index_select(0, competitor)
    normal_norm = torch.linalg.vector_norm(normal, dim=1)
    if bool(normal_norm.le(0.0).any().item()):
        raise BoundaryError("a winner/competitor readout pair has zero normal")
    base_signed = (base_flat * normal).sum(dim=1) / normal_norm
    update_signed = (update_flat * normal).sum(dim=1) / normal_norm
    score_before = base_signed * scale * normal_norm
    score_after = update_signed * scale * normal_norm
    before_error = torch.abs(score_before - rows.transition_margin_before.float())
    after_error = torch.abs(score_after - rows.transition_margin_after.float())
    if float(torch.max(torch.cat((before_error, after_error))).item()) > float(tolerance):
        raise BoundaryError("projected/readout boundary distances do not reproduce score margins")
    displacement = update_signed - base_signed
    return {
        "schema": "token-reconstruction.trr-p12-geometric-boundary.v1",
        "signed_distance_before": base_signed,
        "signed_distance_after": update_signed,
        "signed_distance_displacement": displacement,
        "boundary_normal_norm": normal_norm,
        "projected_movement_norm": torch.linalg.vector_norm(update_flat - base_flat, dim=1),
        "score_margin_reconstruction_error_max": float(torch.max(torch.cat((before_error, after_error))).item()),
        "strict_geometric_crossing": rows.argmax_changed & base_signed.ge(0.0) & update_signed.lt(0.0),
        "interpretation": {
            "distance_units": "projected-hidden Euclidean distance to the hyperplane E[w]-E[c]=0",
            "positive_side": "baseline winner w has larger public-readout score than transition competitor c",
            "displacement": "negative means the projected row moved through the actual pair boundary toward c",
            "projected_movement_norm": "unsigned diagnostic only; it does not establish a crossing",
        },
    }


__all__ = [
    "BOS_TOKEN_ID",
    "BoundaryError",
    "BoundaryRows",
    "SCHEMA",
    "STORED_SEQUENCE_TOKENS",
    "VOCABULARY_SIZE",
    "analyze_score_pair",
    "geometric_boundary_metrics",
    "result_from_compact",
    "top2_full_vocabulary",
]
