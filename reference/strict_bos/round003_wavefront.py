"""Truthless strict-BOS row-serial and pure-wavefront cascade decoders.

The module intentionally depends only on the Round 001 passive teacher types.
It has no token-target row, scorer, evaluator callback, or training entry point.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

import round001_teacher as teacher


class Round003WavefrontError(RuntimeError):
    pass


TIER_K = (32, 128, 512)
TIER_COUNT = len(TIER_K)
FLOAT_EVIDENCE_TOLERANCE = 1e-5
UNEVALUATED_ACCEPT = 255
MAX_CANDIDATE_SEQUENCES_PER_FORWARD = 384


@dataclass(frozen=True)
class CascadeSourceOutput:
    """Fixed-shape result for one 128-row intercepted source."""

    attention_mask: torch.Tensor
    token_ids: torch.Tensor
    route_codes: torch.Tensor
    cascade_confidence: torch.Tensor
    tier_evaluated: torch.Tensor
    tier_winner_token_ids: torch.Tensor
    tier_winner_probability: torch.Tensor
    tier_normalized_winner: torch.Tensor
    tier_score_margin: torch.Tensor
    tier_accepted: torch.Tensor
    candidate_simulations_by_row: torch.Tensor

    def validate(self) -> None:
        matrix = (128, 128)
        tier_shape = (*matrix, TIER_COUNT)
        expected = {
            "attention_mask": (matrix, torch.uint8),
            "token_ids": (matrix, torch.int32),
            "route_codes": (matrix, torch.int8),
            "cascade_confidence": (matrix, torch.float32),
            "tier_evaluated": (tier_shape, torch.uint8),
            "tier_winner_token_ids": (tier_shape, torch.int32),
            "tier_winner_probability": (tier_shape, torch.float32),
            "tier_normalized_winner": (tier_shape, torch.float32),
            "tier_score_margin": (tier_shape, torch.float32),
            "tier_accepted": (tier_shape, torch.uint8),
            "candidate_simulations_by_row": ((128,), torch.int32),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.device.type != "cpu" or value.shape != shape or value.dtype != dtype:
                raise Round003WavefrontError(
                    f"cascade tensor {name} differs from fixed CPU schema"
                )
            if not value.is_contiguous():
                raise Round003WavefrontError(f"cascade tensor {name} is not contiguous")
        mask = self.attention_mask.bool()
        if not torch.logical_or(self.attention_mask.eq(0), self.attention_mask.eq(1)).all():
            raise Round003WavefrontError("attention mask is not binary")
        if self.candidate_simulations_by_row.lt(0).any():
            raise Round003WavefrontError("candidate simulation count is negative")
        if self.route_codes[~mask].ne(teacher.ROUTE_PADDING).any():
            raise Round003WavefrontError("padding carries a non-padding route")
        if self.token_ids[~mask].ne(teacher.INVALID_TOKEN_ID).any():
            raise Round003WavefrontError("padding carries a recovered token")
        evaluated = self.tier_evaluated.bool()
        if not torch.logical_or(self.tier_evaluated.eq(0), self.tier_evaluated.eq(1)).all():
            raise Round003WavefrontError("tier-evaluated tensor is not binary")
        if self.tier_accepted[evaluated].gt(1).any():
            raise Round003WavefrontError("evaluated tier has invalid acceptance code")
        if self.tier_accepted[~evaluated].ne(UNEVALUATED_ACCEPT).any():
            raise Round003WavefrontError("unevaluated tier acceptance sentinel changed")
        if self.tier_winner_token_ids[~evaluated].ne(teacher.INVALID_TOKEN_ID).any():
            raise Round003WavefrontError("unevaluated tier carries a winner")
        for value in (
            self.tier_winner_probability,
            self.tier_normalized_winner,
            self.tier_score_margin,
        ):
            if not torch.isnan(value[~evaluated]).all():
                raise Round003WavefrontError("unevaluated tier float sentinel changed")
            if not torch.isfinite(value[evaluated]).all():
                raise Round003WavefrontError("evaluated tier evidence is non-finite")
        recovered_routes = self.route_codes.eq(teacher.ROUTE_BOS)
        for route in (
            teacher.ROUTE_A1,
            teacher.ROUTE_A2_K32,
            teacher.ROUTE_A2_K128,
            teacher.ROUTE_A2_K512,
        ):
            recovered_routes |= self.route_codes.eq(route)
        if self.token_ids.ge(0).ne(recovered_routes).any():
            raise Round003WavefrontError("route and recovered-token masks disagree")
        if not torch.isfinite(self.cascade_confidence).all():
            raise Round003WavefrontError("cascade confidence is non-finite")
        if (
            self.cascade_confidence.lt(0).any()
            or self.cascade_confidence.gt(1).any()
        ):
            raise Round003WavefrontError("cascade confidence is outside [0,1]")
        if self.cascade_confidence[~recovered_routes].count_nonzero():
            raise Round003WavefrontError("unrecovered position carries confidence")

    def tensors(self, *, prefix: str = "") -> dict[str, torch.Tensor]:
        self.validate()
        return {
            f"{prefix}{name}": getattr(self, name)
            for name in (
                "attention_mask",
                "token_ids",
                "route_codes",
                "cascade_confidence",
                "tier_evaluated",
                "tier_winner_token_ids",
                "tier_winner_probability",
                "tier_normalized_winner",
                "tier_score_margin",
                "tier_accepted",
                "candidate_simulations_by_row",
            )
        }

    @property
    def candidate_simulations(self) -> int:
        return int(self.candidate_simulations_by_row.sum().item())

    @property
    def covered_positions(self) -> int:
        return int(self.token_ids.ge(0).sum().item())


def _blank_output(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    if mask.shape != (128, 128):
        raise Round003WavefrontError("source mask geometry changed")
    mask_u8 = mask.to(device="cpu", dtype=torch.uint8).contiguous()
    return {
        "attention_mask": mask_u8,
        "token_ids": torch.full((128, 128), teacher.INVALID_TOKEN_ID, dtype=torch.int32),
        "route_codes": torch.full(
            (128, 128), teacher.ROUTE_PADDING, dtype=torch.int8
        ),
        "cascade_confidence": torch.zeros((128, 128), dtype=torch.float32),
        "tier_evaluated": torch.zeros((128, 128, TIER_COUNT), dtype=torch.uint8),
        "tier_winner_token_ids": torch.full(
            (128, 128, TIER_COUNT), teacher.INVALID_TOKEN_ID, dtype=torch.int32
        ),
        "tier_winner_probability": torch.full(
            (128, 128, TIER_COUNT), float("nan"), dtype=torch.float32
        ),
        "tier_normalized_winner": torch.full(
            (128, 128, TIER_COUNT), float("nan"), dtype=torch.float32
        ),
        "tier_score_margin": torch.full(
            (128, 128, TIER_COUNT), float("nan"), dtype=torch.float32
        ),
        "tier_accepted": torch.full(
            (128, 128, TIER_COUNT), UNEVALUATED_ACCEPT, dtype=torch.uint8
        ),
        "candidate_simulations_by_row": torch.zeros(128, dtype=torch.int32),
    }


def _finish_output(values: Mapping[str, torch.Tensor]) -> CascadeSourceOutput:
    result = CascadeSourceOutput(
        **{name: value.detach().cpu().contiguous() for name, value in values.items()}
    )
    result.validate()
    return result


def _validate_inputs(
    rows: Sequence[teacher.PassiveRow],
    candidates: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    if len(rows) != 128 or any(row.row_index != index for index, row in enumerate(rows)):
        raise Round003WavefrontError("passive row set/order changed")
    if candidates.shape != (128, 128, 512) or candidates.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise Round003WavefrontError("candidate tensor geometry/dtype changed")
    if confidence.shape != (128, 128) or confidence.dtype != torch.float32:
        raise Round003WavefrontError("A1 confidence geometry/dtype changed")
    mask = torch.stack([row.attention_mask for row in rows]).bool().cpu()
    if not mask.any(dim=1).all():
        raise Round003WavefrontError("a passive row is empty")
    if not torch.isfinite(confidence[mask]).all():
        raise Round003WavefrontError("valid A1 confidence is non-finite")
    if confidence[mask].lt(0).any() or confidence[mask].gt(1).any():
        raise Round003WavefrontError("valid A1 confidence is outside [0,1]")
    return mask


@torch.inference_mode()
def _serial_candidate_hidden(
    precut: teacher.PublicP0Precut,
    *,
    cache: Any,
    candidate_ids: torch.Tensor,
    position: int,
    device: torch.device,
) -> torch.Tensor:
    candidate_cache = copy.deepcopy(cache)
    repeat = getattr(candidate_cache, "batch_repeat_interleave", None)
    if not callable(repeat):
        raise Round003WavefrontError("public cache lacks repeat operation")
    repeat(int(candidate_ids.numel()))
    hidden = precut.run_cached(
        candidate_ids.to(device=device, dtype=torch.long).view(-1, 1),
        candidate_cache,
        position,
    )
    return hidden[:, -1].float()


def _centered_scores(
    hidden: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden.ndim != 2 or target.ndim != 1 or hidden.shape[1] != target.numel():
        raise Round003WavefrontError("serial score geometry changed")
    mean = hidden.float().mean(dim=0, keepdim=True)
    scores = F.cosine_similarity(
        hidden.float() - mean, target.float().view(1, -1) - mean, dim=-1
    )
    posterior = torch.softmax(scores.float(), dim=0)
    winner = scores.argmax()
    probability = posterior[winner]
    normalized = hidden.shape[0] * probability
    best_two = torch.topk(scores, k=2).values
    margin = best_two[0] - best_two[1]
    return winner, probability, normalized, margin


@torch.inference_mode()
def decode_row_serial_source(
    rows: Sequence[teacher.PassiveRow],
    *,
    candidates: torch.Tensor,
    a1_confidence: torch.Tensor,
    precut: teacher.PublicP0Precut,
    device: torch.device,
) -> CascadeSourceOutput:
    """Reference decoder with one authoritative cache per row."""

    mask = _validate_inputs(rows, candidates, a1_confidence)
    values = _blank_output(mask)
    for row_index, row in enumerate(rows):
        valid = torch.nonzero(mask[row_index], as_tuple=False).flatten().tolist()
        first = valid[0]
        values["token_ids"][row_index, first] = teacher.BOS_TOKEN_ID
        values["route_codes"][row_index, first] = teacher.ROUTE_BOS
        values["cascade_confidence"][row_index, first] = 1.0
        cache = precut.new_cache()
        precut.run_cached(
            torch.tensor([[teacher.BOS_TOKEN_ID]], device=device, dtype=torch.long),
            cache,
            0,
        )
        stopped = False
        for logical, physical in enumerate(valid[1:], start=1):
            if stopped:
                values["route_codes"][row_index, physical] = (
                    teacher.ROUTE_ABSTAINED_SUFFIX
                )
                continue
            confidence = float(a1_confidence[row_index, physical].item())
            if confidence >= teacher.A1_FAST_PATH_MIN_CONFIDENCE:
                chosen = int(candidates[row_index, physical, 0].item())
                route = teacher.ROUTE_A1
                chosen_confidence = confidence
            else:
                parts: list[torch.Tensor] = []
                previous = 0
                chosen = teacher.INVALID_TOKEN_ID
                route = teacher.ROUTE_ABSTAIN_K512
                chosen_confidence = 0.0
                for tier_index, k in enumerate(TIER_K):
                    parts.append(
                        _serial_candidate_hidden(
                            precut,
                            cache=cache,
                            candidate_ids=candidates[
                                row_index, physical, previous:k
                            ],
                            position=logical,
                            device=device,
                        )
                    )
                    hidden = torch.cat(parts, dim=0)
                    winner, probability, normalized, margin = _centered_scores(
                        hidden, row.activation[physical].to(device).float()
                    )
                    winner_token = int(
                        candidates[row_index, physical, int(winner.item())].item()
                    )
                    accepted = bool(
                        normalized.item() >= teacher.NORMALIZED_WINNER_MIN
                    )
                    values["tier_evaluated"][row_index, physical, tier_index] = 1
                    values["tier_winner_token_ids"][
                        row_index, physical, tier_index
                    ] = winner_token
                    values["tier_winner_probability"][
                        row_index, physical, tier_index
                    ] = float(probability.item())
                    values["tier_normalized_winner"][
                        row_index, physical, tier_index
                    ] = float(normalized.item())
                    values["tier_score_margin"][
                        row_index, physical, tier_index
                    ] = float(margin.item())
                    values["tier_accepted"][row_index, physical, tier_index] = int(
                        accepted
                    )
                    values["candidate_simulations_by_row"][row_index] += k - previous
                    if accepted:
                        chosen = winner_token
                        route = {
                            32: teacher.ROUTE_A2_K32,
                            128: teacher.ROUTE_A2_K128,
                            512: teacher.ROUTE_A2_K512,
                        }[k]
                        chosen_confidence = float(probability.item())
                        break
                    previous = k
                if chosen == teacher.INVALID_TOKEN_ID:
                    values["route_codes"][row_index, physical] = (
                        teacher.ROUTE_ABSTAIN_K512
                    )
                    stopped = True
                    continue
            values["token_ids"][row_index, physical] = chosen
            values["route_codes"][row_index, physical] = route
            values["cascade_confidence"][row_index, physical] = chosen_confidence
            precut.run_cached(
                torch.tensor([[chosen]], device=device, dtype=torch.long),
                cache,
                logical,
            )
    return _finish_output(values)


@torch.inference_mode()
def _batched_candidate_hidden(
    precut: teacher.PublicP0Precut,
    *,
    cache: Any,
    parent_indices: torch.Tensor,
    candidate_ids: torch.Tensor,
    position: int,
) -> torch.Tensor:
    if parent_indices.ndim != 1 or candidate_ids.ndim != 2:
        raise Round003WavefrontError("batched candidate geometry changed")
    parents, width = candidate_ids.shape
    if parents != parent_indices.numel() or parents == 0 or width == 0:
        raise Round003WavefrontError("batched candidate set is empty or inconsistent")
    if width > MAX_CANDIDATE_SEQUENCES_PER_FORWARD:
        raise Round003WavefrontError(
            "candidate tier exceeds the frozen batch limit"
        )
    parents_per_forward = max(
        1, MAX_CANDIDATE_SEQUENCES_PER_FORWARD // width
    )
    blocks: list[torch.Tensor] = []
    for start in range(0, parents, parents_per_forward):
        stop = min(parents, start + parents_per_forward)
        candidate_cache = copy.deepcopy(cache)
        candidate_cache.batch_select_indices(parent_indices[start:stop])
        candidate_cache.batch_repeat_interleave(width)
        hidden = precut.run_cached(
            candidate_ids[start:stop].reshape(-1, 1),
            candidate_cache,
            position,
        )
        blocks.append(
            hidden[:, -1].float().reshape(stop - start, width, -1)
        )
        del candidate_cache, hidden
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)


def _batched_scores(
    hidden: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or target.ndim != 2 or hidden.shape[0] != target.shape[0]:
        raise Round003WavefrontError("batched score geometry changed")
    mean = hidden.float().mean(dim=1, keepdim=True)
    scores = F.cosine_similarity(
        hidden.float() - mean,
        target.float().unsqueeze(1) - mean,
        dim=-1,
    )
    posterior = torch.softmax(scores.float(), dim=-1)
    winner = scores.argmax(dim=-1)
    probability = posterior.gather(1, winner.view(-1, 1)).squeeze(1)
    normalized = hidden.shape[1] * probability
    best_two = torch.topk(scores, k=2, dim=-1).values
    margin = best_two[:, 0] - best_two[:, 1]
    return winner, probability, normalized, margin


@torch.inference_mode()
def decode_wavefront_source(
    rows: Sequence[teacher.PassiveRow],
    *,
    candidates: torch.Tensor,
    a1_confidence: torch.Tensor,
    precut: teacher.PublicP0Precut,
    device: torch.device,
) -> CascadeSourceOutput:
    """Pure wavefront decode with batching only across independent rows."""

    mask = _validate_inputs(rows, candidates, a1_confidence)
    values = _blank_output(mask)
    valid_by_row = [
        torch.nonzero(mask[index], as_tuple=False).flatten().tolist()
        for index in range(128)
    ]
    for row_index, valid in enumerate(valid_by_row):
        first = valid[0]
        values["token_ids"][row_index, first] = teacher.BOS_TOKEN_ID
        values["route_codes"][row_index, first] = teacher.ROUTE_BOS
        values["cascade_confidence"][row_index, first] = 1.0

    active_rows = list(range(128))
    cache = precut.new_cache()
    precut.run_cached(
        torch.full((128, 1), teacher.BOS_TOKEN_ID, device=device, dtype=torch.long),
        cache,
        0,
    )
    max_length = max(len(valid) for valid in valid_by_row)
    for logical in range(1, max_length):
        keep_ended = [
            local
            for local, row_index in enumerate(active_rows)
            if len(valid_by_row[row_index]) > logical
        ]
        if len(keep_ended) != len(active_rows):
            cache.batch_select_indices(
                torch.tensor(keep_ended, device=device, dtype=torch.long)
            )
            active_rows = [active_rows[local] for local in keep_ended]
        if not active_rows:
            break

        physical = [valid_by_row[row_index][logical] for row_index in active_rows]
        proposals = torch.stack(
            [
                candidates[row_index, physical_position]
                for row_index, physical_position in zip(active_rows, physical, strict=True)
            ]
        ).to(device=device, dtype=torch.long)
        a1_probability = torch.tensor(
            [
                float(a1_confidence[row_index, physical_position].item())
                for row_index, physical_position in zip(active_rows, physical, strict=True)
            ],
            device=device,
            dtype=torch.float32,
        )
        targets = torch.stack(
            [
                rows[row_index].activation[physical_position].float()
                for row_index, physical_position in zip(active_rows, physical, strict=True)
            ]
        ).to(device)
        batch = len(active_rows)
        chosen = torch.full(
            (batch,), teacher.INVALID_TOKEN_ID, device=device, dtype=torch.long
        )
        chosen_probability = torch.zeros(batch, device=device, dtype=torch.float32)
        route_here = torch.full(
            (batch,), teacher.ROUTE_PADDING, device=device, dtype=torch.int8
        )

        fast_indices = torch.nonzero(
            a1_probability.ge(teacher.A1_FAST_PATH_MIN_CONFIDENCE),
            as_tuple=False,
        ).flatten()
        if fast_indices.numel():
            chosen[fast_indices] = proposals[fast_indices, 0]
            chosen_probability[fast_indices] = a1_probability[fast_indices]
            route_here[fast_indices] = teacher.ROUTE_A1

        pending_active = torch.nonzero(
            chosen.eq(teacher.INVALID_TOKEN_ID), as_tuple=False
        ).flatten()
        pending_history: torch.Tensor | None = None
        previous = 0
        for tier_index, k in enumerate(TIER_K):
            if not pending_active.numel():
                break
            width = k - previous
            segment = _batched_candidate_hidden(
                precut,
                cache=cache,
                parent_indices=pending_active,
                candidate_ids=proposals[pending_active, previous:k],
                position=logical,
            )
            pending_history = (
                segment
                if pending_history is None
                else torch.cat((pending_history, segment), dim=1)
            )
            winner, probability, normalized, margin = _batched_scores(
                pending_history, targets[pending_active]
            )
            winner_token = proposals[pending_active].gather(
                1, winner.view(-1, 1)
            ).squeeze(1)
            accepted = normalized.ge(teacher.NORMALIZED_WINNER_MIN)
            for offset, active_index in enumerate(pending_active.tolist()):
                row_index = active_rows[active_index]
                physical_position = physical[active_index]
                values["tier_evaluated"][
                    row_index, physical_position, tier_index
                ] = 1
                values["tier_winner_token_ids"][
                    row_index, physical_position, tier_index
                ] = int(winner_token[offset].item())
                values["tier_winner_probability"][
                    row_index, physical_position, tier_index
                ] = float(probability[offset].item())
                values["tier_normalized_winner"][
                    row_index, physical_position, tier_index
                ] = float(normalized[offset].item())
                values["tier_score_margin"][
                    row_index, physical_position, tier_index
                ] = float(margin[offset].item())
                values["tier_accepted"][
                    row_index, physical_position, tier_index
                ] = int(accepted[offset].item())
                values["candidate_simulations_by_row"][row_index] += width
            accepted_offsets = torch.nonzero(accepted, as_tuple=False).flatten()
            if accepted_offsets.numel():
                accepted_active = pending_active[accepted_offsets]
                chosen[accepted_active] = winner_token[accepted_offsets]
                chosen_probability[accepted_active] = probability[accepted_offsets]
                route_here[accepted_active] = {
                    32: teacher.ROUTE_A2_K32,
                    128: teacher.ROUTE_A2_K128,
                    512: teacher.ROUTE_A2_K512,
                }[k]
            remaining = torch.nonzero(~accepted, as_tuple=False).flatten()
            pending_active = pending_active[remaining]
            pending_history = pending_history[remaining]
            previous = k

        abstained = torch.nonzero(
            chosen.eq(teacher.INVALID_TOKEN_ID), as_tuple=False
        ).flatten()
        abstained_set = set(abstained.tolist())
        for local, row_index in enumerate(active_rows):
            physical_position = physical[local]
            if local in abstained_set:
                values["route_codes"][row_index, physical_position] = (
                    teacher.ROUTE_ABSTAIN_K512
                )
                for suffix_logical in range(logical + 1, len(valid_by_row[row_index])):
                    suffix_physical = valid_by_row[row_index][suffix_logical]
                    values["route_codes"][row_index, suffix_physical] = (
                        teacher.ROUTE_ABSTAINED_SUFFIX
                    )
                continue
            token = int(chosen[local].item())
            values["token_ids"][row_index, physical_position] = token
            values["route_codes"][row_index, physical_position] = int(
                route_here[local].item()
            )
            values["cascade_confidence"][row_index, physical_position] = float(
                chosen_probability[local].item()
            )

        commit = torch.nonzero(
            chosen.ne(teacher.INVALID_TOKEN_ID), as_tuple=False
        ).flatten()
        if not commit.numel():
            active_rows = []
            break
        cache.batch_select_indices(commit)
        precut.run_cached(chosen[commit].view(-1, 1), cache, logical)
        active_rows = [active_rows[index] for index in commit.tolist()]
    return _finish_output(values)


def _nonfinite_categories(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(value.shape, dtype=torch.uint8)
    result[torch.isnan(value)] = 1
    result[torch.isposinf(value)] = 2
    result[torch.isneginf(value)] = 3
    return result


def compare_outputs(
    reference: CascadeSourceOutput,
    candidate: CascadeSourceOutput,
    *,
    float_tolerance: float = FLOAT_EVIDENCE_TOLERANCE,
) -> dict[str, Any]:
    """Fail-closed semantic comparison, including non-finite categories."""

    reference.validate()
    candidate.validate()
    if float_tolerance != FLOAT_EVIDENCE_TOLERANCE:
        raise Round003WavefrontError("wavefront comparison tolerance changed")
    exact_names = (
        "attention_mask",
        "token_ids",
        "route_codes",
        "tier_evaluated",
        "tier_winner_token_ids",
        "tier_accepted",
        "candidate_simulations_by_row",
    )
    for name in exact_names:
        if not torch.equal(getattr(reference, name), getattr(candidate, name)):
            raise Round003WavefrontError(f"row-serial/wavefront mismatch: {name}")
    maximums: dict[str, float] = {}
    for name in (
        "cascade_confidence",
        "tier_winner_probability",
        "tier_normalized_winner",
        "tier_score_margin",
    ):
        lhs = getattr(reference, name)
        rhs = getattr(candidate, name)
        lhs_category = _nonfinite_categories(lhs)
        rhs_category = _nonfinite_categories(rhs)
        if not torch.equal(lhs_category, rhs_category):
            raise Round003WavefrontError(
                f"row-serial/wavefront non-finite category mismatch: {name}"
            )
        finite = lhs_category.eq(0)
        maximum = (
            0.0
            if not finite.any()
            else float((lhs[finite] - rhs[finite]).abs().max().item())
        )
        if not math.isfinite(maximum) or maximum > float_tolerance:
            raise Round003WavefrontError(
                f"row-serial/wavefront finite mismatch: {name}={maximum}"
            )
        maximums[name] = maximum
    return {
        "all_integer_boolean_and_nonfinite_categories_exact": True,
        "finite_float_max_absolute_difference": maximums,
        "finite_float_tolerance": float_tolerance,
        "covered_positions": reference.covered_positions,
        "candidate_simulations": reference.candidate_simulations,
    }


def output_from_tensors(
    tensors: Mapping[str, torch.Tensor], *, prefix: str = ""
) -> CascadeSourceOutput:
    names = (
        "attention_mask",
        "token_ids",
        "route_codes",
        "cascade_confidence",
        "tier_evaluated",
        "tier_winner_token_ids",
        "tier_winner_probability",
        "tier_normalized_winner",
        "tier_score_margin",
        "tier_accepted",
        "candidate_simulations_by_row",
    )
    expected = {f"{prefix}{name}" for name in names}
    if not expected.issubset(tensors):
        raise Round003WavefrontError("cascade tensor mapping is incomplete")
    return _finish_output({name: tensors[f"{prefix}{name}"] for name in names})


def stack_outputs(
    source_indices: Sequence[int], outputs: Sequence[CascadeSourceOutput]
) -> dict[str, torch.Tensor]:
    if not source_indices or len(source_indices) != len(outputs):
        raise Round003WavefrontError("source/output stack is empty or mismatched")
    for output in outputs:
        output.validate()
    result: dict[str, torch.Tensor] = {
        "source_indices": torch.tensor(source_indices, dtype=torch.int64)
    }
    for name in outputs[0].tensors():
        result[name] = torch.stack([getattr(output, name) for output in outputs])
    return {name: value.contiguous() for name, value in result.items()}


__all__ = [
    "CascadeSourceOutput",
    "FLOAT_EVIDENCE_TOLERANCE",
    "Round003WavefrontError",
    "TIER_COUNT",
    "TIER_K",
    "compare_outputs",
    "decode_row_serial_source",
    "decode_wavefront_source",
    "output_from_tensors",
    "stack_outputs",
]
