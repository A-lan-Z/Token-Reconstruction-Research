"""Exhaustive bounded A1+A2 configuration-search primitives.

The public A1 proposer is deliberately outside this module.  A policy receives
its frozen candidate ordering and may vary only causal simulation, routing, and
terminal behavior as preregistered for TRR-0002 owner revision R1.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import time
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn.functional as F

from .dual_benchmark import (
    BOS_TOKEN_ID,
    INVALID_TOKEN_ID,
    validate_observations,
)


BUDGET_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
ADAPTIVE_GRID = (2, 4, 8, 16, 32, 64, 128, 256, 512)
SCORE_RULES = ("direct_cosine", "group_centered_cosine")
FAST_PATHS: tuple[tuple[str, float | None], ...] = (
    ("off", None),
    ("a1_ge_0.90", 0.90),
    ("a1_ge_0.95", 0.95),
    ("a1_ge_0.99", 0.99),
    ("historical_a1_ge_0.999", 0.999),
)
ROUTING_SIGNALS = (
    "normalized_softmax_winner",
    "raw_margin",
    "rms_normalized_margin",
)
EMPIRICAL_GATE_MODES = (
    "always_accept",
    "fit_quantile_0.05",
    "fit_quantile_0.10",
    "fit_quantile_0.20",
    "fit_quantile_0.35",
    "fit_quantile_0.50",
    "fit_quantile_0.65",
    "fit_quantile_0.80",
    "fit_quantile_0.90",
    "fit_quantile_0.95",
    "never_accept",
)
TERMINAL_ACTIONS = (
    "commit_last_winner",
    "fallback_to_a1",
    "abstain_and_stop_suffix",
)
HISTORICAL_GATE_MODE = "historical_absolute_2.0"
CURRENT_GATE_MODE = "current_absolute_1.2544946670532227"
CURRENT_THRESHOLD = 1.2544946670532227
MAX_CANDIDATE_SEQUENCES_PER_FORWARD = 384

ROUTE_PADDING = 0
ROUTE_BOS = 1
ROUTE_FAST_A1 = 2
ROUTE_TIER = 3
ROUTE_FINAL_A1_FALLBACK = 4
ROUTE_ABSTAIN = 5
ROUTE_ABSTAINED_SUFFIX = 6


class ConfigurationSearchError(RuntimeError):
    """Raised when a frozen configuration-search contract is violated."""


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    score_rule: str
    schedule: tuple[int, ...]
    fast_path_id: str
    fast_path_threshold: float | None
    routing_signal: str | None
    gate_mode: str | None
    terminal_action: str
    gate_comparator: str = "ge"

    def validate(self) -> None:
        if self.kind not in {"fixed", "adaptive", "current_anchor"}:
            raise ConfigurationSearchError("unknown policy kind")
        if self.score_rule not in SCORE_RULES:
            raise ConfigurationSearchError("unknown score rule")
        if not self.schedule or any(value not in BUDGET_GRID for value in self.schedule):
            raise ConfigurationSearchError("policy schedule leaves the frozen grid")
        if tuple(sorted(set(self.schedule))) != self.schedule:
            raise ConfigurationSearchError("policy schedule is not strictly increasing")
        expected_fast = dict(FAST_PATHS).get(self.fast_path_id, object())
        if expected_fast != self.fast_path_threshold:
            raise ConfigurationSearchError("fast-path ID and threshold disagree")
        if self.gate_comparator not in {"ge", "gt"}:
            raise ConfigurationSearchError("unknown gate comparator")
        if self.kind == "fixed":
            if len(self.schedule) != 1 or self.routing_signal is not None:
                raise ConfigurationSearchError("fixed policy has adaptive fields")
            if self.gate_mode is not None or self.terminal_action != "commit_last_winner":
                raise ConfigurationSearchError("fixed policy has a gate or terminal variant")
            if self.score_rule == "group_centered_cosine" and self.schedule == (1,):
                raise ConfigurationSearchError("centered K1 is undefined")
            return
        if len(self.schedule) < 2 or self.schedule[0] < 2:
            raise ConfigurationSearchError("adaptive schedule is too short or starts at K1")
        if self.routing_signal not in ROUTING_SIGNALS:
            raise ConfigurationSearchError("adaptive policy lacks a routing signal")
        if self.terminal_action not in TERMINAL_ACTIONS:
            raise ConfigurationSearchError("unknown terminal action")
        valid_gates = {*EMPIRICAL_GATE_MODES, HISTORICAL_GATE_MODE, CURRENT_GATE_MODE}
        if self.gate_mode not in valid_gates:
            raise ConfigurationSearchError("unknown adaptive gate")
        if self.kind == "current_anchor":
            expected = (
                self.score_rule == "direct_cosine"
                and self.schedule == (32, 64)
                and self.fast_path_threshold is None
                and self.routing_signal == "rms_normalized_margin"
                and self.gate_mode == CURRENT_GATE_MODE
                and self.terminal_action == "commit_last_winner"
                and self.gate_comparator == "gt"
            )
            if not expected:
                raise ConfigurationSearchError("current anchor changed")

    def serialized(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["schedule"] = list(self.schedule)
        return value

    @property
    def policy_id(self) -> str:
        payload = json.dumps(
            self.serialized(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "a1a2_" + hashlib.sha256(payload).hexdigest()[:20]


@dataclass(frozen=True)
class ResolvedPolicy:
    spec: PolicySpec
    thresholds: tuple[tuple[int, float], ...]

    def validate(self) -> None:
        self.spec.validate()
        if self.spec.kind == "fixed":
            if self.thresholds:
                raise ConfigurationSearchError("fixed policy carries thresholds")
            return
        keys = tuple(value for value, _ in self.thresholds)
        if keys != self.spec.schedule:
            raise ConfigurationSearchError("resolved threshold grid changed")
        if any(math.isnan(value) for _, value in self.thresholds):
            raise ConfigurationSearchError("resolved threshold is NaN")

    @property
    def policy_id(self) -> str:
        return self.spec.policy_id

    def threshold_at(self, k: int) -> float:
        for budget, threshold in self.thresholds:
            if budget == k:
                return threshold
        raise ConfigurationSearchError(f"policy has no threshold at K={k}")

    def serialized(self) -> dict[str, Any]:
        self.validate()
        return {
            "policy_id": self.policy_id,
            "spec": self.spec.serialized(),
            "numeric_thresholds": {
                str(k): (
                    "-Infinity"
                    if value == float("-inf")
                    else "Infinity"
                    if value == float("inf")
                    else value
                )
                for k, value in self.thresholds
            },
        }


def policy_spec_from_dict(value: Mapping[str, Any]) -> PolicySpec:
    spec = PolicySpec(
        kind=str(value["kind"]),
        score_rule=str(value["score_rule"]),
        schedule=tuple(int(k) for k in value["schedule"]),
        fast_path_id=str(value["fast_path_id"]),
        fast_path_threshold=(
            None
            if value["fast_path_threshold"] is None
            else float(value["fast_path_threshold"])
        ),
        routing_signal=(
            None if value["routing_signal"] is None else str(value["routing_signal"])
        ),
        gate_mode=None if value["gate_mode"] is None else str(value["gate_mode"]),
        terminal_action=str(value["terminal_action"]),
        gate_comparator=str(value.get("gate_comparator", "ge")),
    )
    spec.validate()
    return spec


def resolved_policy_from_dict(value: Mapping[str, Any]) -> ResolvedPolicy:
    spec = policy_spec_from_dict(value["spec"])
    encoded = value["numeric_thresholds"]

    def decode_threshold(raw: Any) -> float:
        if raw == "-Infinity":
            return float("-inf")
        if raw == "Infinity":
            return float("inf")
        return float(raw)

    policy = ResolvedPolicy(
        spec=spec,
        thresholds=tuple(
            (k, decode_threshold(encoded[str(k)])) for k in spec.schedule
        ),
    )
    policy.validate()
    if value.get("policy_id") != policy.policy_id:
        raise ConfigurationSearchError("serialized policy ID changed")
    return policy


@dataclass(frozen=True)
class DecodeResult:
    predictions: torch.Tensor
    routes: torch.Tensor
    selected_k: torch.Tensor
    selected_signal: torch.Tensor
    elapsed_seconds: float
    candidate_simulations: int
    executed_candidate_simulations: int
    prefix_commit_tokens: int
    record_batch_size: int


def adaptive_schedules() -> tuple[tuple[int, ...], ...]:
    return tuple(
        schedule
        for length in range(2, len(ADAPTIVE_GRID) + 1)
        for schedule in itertools.combinations(ADAPTIVE_GRID, length)
    )


def declared_policy_count() -> int:
    fixed = (len(BUDGET_GRID) + len(BUDGET_GRID) - 1) * len(FAST_PATHS)
    schedules = len(adaptive_schedules())
    empirical = (
        schedules
        * len(SCORE_RULES)
        * len(FAST_PATHS)
        * len(ROUTING_SIGNALS)
        * len(EMPIRICAL_GATE_MODES)
        * len(TERMINAL_ACTIONS)
    )
    historical = (
        schedules * len(SCORE_RULES) * len(FAST_PATHS) * len(TERMINAL_ACTIONS)
    )
    return fixed + empirical + historical + 1


def current_anchor_spec() -> PolicySpec:
    value = PolicySpec(
        kind="current_anchor",
        score_rule="direct_cosine",
        schedule=(32, 64),
        fast_path_id="off",
        fast_path_threshold=None,
        routing_signal="rms_normalized_margin",
        gate_mode=CURRENT_GATE_MODE,
        terminal_action="commit_last_winner",
        gate_comparator="gt",
    )
    value.validate()
    return value


def historical_anchor_spec() -> PolicySpec:
    value = PolicySpec(
        kind="adaptive",
        score_rule="group_centered_cosine",
        schedule=(32, 128, 512),
        fast_path_id="historical_a1_ge_0.999",
        fast_path_threshold=0.999,
        routing_signal="normalized_softmax_winner",
        gate_mode=HISTORICAL_GATE_MODE,
        terminal_action="abstain_and_stop_suffix",
    )
    value.validate()
    return value


def iter_policy_specs() -> Iterator[PolicySpec]:
    for score_rule in SCORE_RULES:
        for budget in BUDGET_GRID:
            if score_rule == "group_centered_cosine" and budget == 1:
                continue
            for fast_id, fast_threshold in FAST_PATHS:
                yield PolicySpec(
                    kind="fixed",
                    score_rule=score_rule,
                    schedule=(budget,),
                    fast_path_id=fast_id,
                    fast_path_threshold=fast_threshold,
                    routing_signal=None,
                    gate_mode=None,
                    terminal_action="commit_last_winner",
                )
    schedules = adaptive_schedules()
    for schedule in schedules:
        for score_rule in SCORE_RULES:
            for fast_id, fast_threshold in FAST_PATHS:
                for signal in ROUTING_SIGNALS:
                    for gate_mode in EMPIRICAL_GATE_MODES:
                        for terminal in TERMINAL_ACTIONS:
                            yield PolicySpec(
                                kind="adaptive",
                                score_rule=score_rule,
                                schedule=schedule,
                                fast_path_id=fast_id,
                                fast_path_threshold=fast_threshold,
                                routing_signal=signal,
                                gate_mode=gate_mode,
                                terminal_action=terminal,
                            )
                for terminal in TERMINAL_ACTIONS:
                    yield PolicySpec(
                        kind="adaptive",
                        score_rule=score_rule,
                        schedule=schedule,
                        fast_path_id=fast_id,
                        fast_path_threshold=fast_threshold,
                        routing_signal="normalized_softmax_winner",
                        gate_mode=HISTORICAL_GATE_MODE,
                        terminal_action=terminal,
                    )
    yield current_anchor_spec()


def resolve_policy(
    spec: PolicySpec,
    fitted_thresholds: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> ResolvedPolicy:
    spec.validate()
    if spec.kind == "fixed":
        value = ResolvedPolicy(spec=spec, thresholds=())
        value.validate()
        return value
    if spec.gate_mode == CURRENT_GATE_MODE:
        thresholds = tuple((k, CURRENT_THRESHOLD) for k in spec.schedule)
    elif spec.gate_mode == HISTORICAL_GATE_MODE:
        thresholds = tuple((k, 2.0) for k in spec.schedule)
    elif spec.gate_mode == "always_accept":
        thresholds = tuple((k, float("-inf")) for k in spec.schedule)
    elif spec.gate_mode == "never_accept":
        thresholds = tuple((k, float("inf")) for k in spec.schedule)
    else:
        if spec.routing_signal is None or spec.gate_mode is None:
            raise ConfigurationSearchError("adaptive policy is missing fitted gate fields")
        try:
            thresholds = tuple(
                (
                    k,
                    float(
                        fitted_thresholds[spec.score_rule][spec.routing_signal][str(k)][
                            spec.gate_mode
                        ]
                    ),
                )
                for k in spec.schedule
            )
        except KeyError as exc:
            raise ConfigurationSearchError(
                f"missing fitted threshold for {spec.policy_id}"
            ) from exc
    value = ResolvedPolicy(spec=spec, thresholds=thresholds)
    value.validate()
    return value


def score_candidates(
    hidden: torch.Tensor,
    target: torch.Tensor,
    score_rule: str,
) -> torch.Tensor:
    if hidden.ndim != 3 or target.ndim != 2:
        raise ConfigurationSearchError("candidate score geometry changed")
    if hidden.shape[0] != target.shape[0] or hidden.shape[2] != target.shape[1]:
        raise ConfigurationSearchError("candidate and target geometry disagree")
    if score_rule == "direct_cosine":
        scores = F.cosine_similarity(hidden.float(), target[:, None, :].float(), dim=-1)
    elif score_rule == "group_centered_cosine":
        mean = hidden.float().mean(dim=1, keepdim=True)
        scores = F.cosine_similarity(
            hidden.float() - mean,
            target[:, None, :].float() - mean,
            dim=-1,
        )
    else:
        raise ConfigurationSearchError("unknown candidate score rule")
    if not torch.isfinite(scores).all().item():
        raise ConfigurationSearchError("candidate scores are non-finite")
    return scores


def routing_signal(scores: torch.Tensor, signal: str) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ConfigurationSearchError("routing signal needs at least two candidates")
    best_two = torch.topk(scores.float(), k=2, dim=1).values
    margin = best_two[:, 0] - best_two[:, 1]
    if signal == "normalized_softmax_winner":
        value = scores.shape[1] * torch.softmax(scores.float(), dim=1).max(dim=1).values
    elif signal == "raw_margin":
        value = margin
    elif signal == "rms_normalized_margin":
        centered = scores.float() - scores.float().mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1).sqrt().clamp_min(1e-8)
        value = margin / scale
    else:
        raise ConfigurationSearchError("unknown routing signal")
    if not torch.isfinite(value).all().item() or value.lt(0).any().item():
        raise ConfigurationSearchError("routing signal is invalid")
    return value


def gate_passes(value: torch.Tensor, threshold: float, comparator: str) -> torch.Tensor:
    if comparator == "ge":
        return value.ge(threshold)
    if comparator == "gt":
        return value.gt(threshold)
    raise ConfigurationSearchError("unknown gate comparator")


def _candidate_hidden(
    precut: torch.nn.Module,
    *,
    cache: Any,
    parent_indices: torch.Tensor,
    candidate_ids: torch.Tensor,
    position: int,
) -> torch.Tensor:
    if parent_indices.ndim != 1 or candidate_ids.ndim != 2:
        raise ConfigurationSearchError("batched candidate geometry changed")
    parents, width = candidate_ids.shape
    if parents != parent_indices.numel() or parents == 0 or width == 0:
        raise ConfigurationSearchError("empty or inconsistent candidate batch")
    candidate_parts: list[torch.Tensor] = []
    for candidate_start in range(0, width, MAX_CANDIDATE_SEQUENCES_PER_FORWARD):
        candidate_stop = min(
            width, candidate_start + MAX_CANDIDATE_SEQUENCES_PER_FORWARD
        )
        candidate_width = candidate_stop - candidate_start
        parents_per_forward = max(
            1, MAX_CANDIDATE_SEQUENCES_PER_FORWARD // candidate_width
        )
        parent_parts: list[torch.Tensor] = []
        for parent_start in range(0, parents, parents_per_forward):
            parent_stop = min(parents, parent_start + parents_per_forward)
            candidate_cache = copy.deepcopy(cache)
            selector = getattr(candidate_cache, "batch_select_indices", None)
            repeat = getattr(candidate_cache, "batch_repeat_interleave", None)
            if not callable(selector) or not callable(repeat):
                raise ConfigurationSearchError("public cache lacks batch operations")
            selector(parent_indices[parent_start:parent_stop])
            repeat(candidate_width)
            ids = candidate_ids[
                parent_start:parent_stop, candidate_start:candidate_stop
            ]
            hidden = precut.run_cached(
                ids.reshape(-1, 1), candidate_cache, position
            )[:, -1].float().reshape(parent_stop - parent_start, candidate_width, -1)
            parent_parts.append(hidden)
            del candidate_cache
        candidate_parts.append(
            parent_parts[0] if len(parent_parts) == 1 else torch.cat(parent_parts, dim=0)
        )
    return (
        candidate_parts[0]
        if len(candidate_parts) == 1
        else torch.cat(candidate_parts, dim=1)
    )


@torch.inference_mode()
def decode_policy(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    candidates: torch.Tensor,
    a1_confidence: torch.Tensor,
    precut: torch.nn.Module,
    device: torch.device,
    policy: ResolvedPolicy,
    record_batch_size: int | None = None,
) -> DecodeResult:
    """Run one resolved policy with a strictly reconstructed causal prefix."""

    policy.validate()
    validate_observations(observations, attention_mask, position_ids)
    if candidates.shape[:2] != attention_mask.shape or candidates.ndim != 3:
        raise ConfigurationSearchError("candidate geometry differs from observations")
    if candidates.shape[2] < policy.spec.schedule[-1]:
        raise ConfigurationSearchError("candidate list is shallower than policy")
    if a1_confidence.shape != attention_mask.shape:
        raise ConfigurationSearchError("A1 confidence geometry changed")
    valid = attention_mask.to(torch.bool)
    confidence_mask = valid.clone()
    confidence_mask[:, 0] = False
    if not torch.isfinite(a1_confidence[confidence_mask]).all().item():
        raise ConfigurationSearchError("valid A1 confidence is non-finite")
    if a1_confidence[confidence_mask].lt(0).any().item() or a1_confidence[confidence_mask].gt(1).any().item():
        raise ConfigurationSearchError("valid A1 confidence is outside [0,1]")
    if record_batch_size is None:
        record_batch_size = 8
    if not 0 < record_batch_size <= 16:
        raise ConfigurationSearchError("record batch size is invalid")

    predictions = torch.full(attention_mask.shape, INVALID_TOKEN_ID, dtype=torch.long)
    predictions[:, 0] = BOS_TOKEN_ID
    routes = torch.full(attention_mask.shape, ROUTE_PADDING, dtype=torch.int8)
    routes[:, 0] = ROUTE_BOS
    selected_k = torch.zeros(attention_mask.shape, dtype=torch.int16)
    selected_signal = torch.full(attention_mask.shape, float("nan"), dtype=torch.float32)
    logical_simulations = 0
    executed_simulations = 0
    prefix_commit_tokens = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for record_start in range(0, observations.shape[0], record_batch_size):
        record_end = min(record_start + record_batch_size, observations.shape[0])
        count = record_end - record_start
        batch_mask = attention_mask[record_start:record_end].to(torch.bool)
        stopped = torch.zeros(count, dtype=torch.bool)
        maximum_length = int(batch_mask.sum(dim=1).max().item())
        cache = precut.new_cache()
        precut.run_cached(
            torch.full((count, 1), BOS_TOKEN_ID, dtype=torch.long, device=device),
            cache,
            0,
        )
        prefix_commit_tokens += count

        for position in range(1, maximum_length):
            active_cpu = batch_mask[:, position]
            suffix_cpu = active_cpu & stopped
            routes[record_start:record_end, position][suffix_cpu] = ROUTE_ABSTAINED_SUFFIX
            eligible_cpu = active_cpu & ~stopped
            commit = torch.full(
                (count,), BOS_TOKEN_ID, dtype=torch.long, device=device
            )
            if not eligible_cpu.any().item():
                precut.run_cached(commit[:, None], cache, position)
                prefix_commit_tokens += count
                continue

            eligible = eligible_cpu.to(device)
            ids = candidates[record_start:record_end, position].to(
                device=device, dtype=torch.long
            )
            if ids[eligible].lt(0).any().item():
                raise ConfigurationSearchError("eligible candidate row contains invalid IDs")
            target = observations[record_start:record_end, position].to(device).float()
            confidence = a1_confidence[record_start:record_end, position].to(device)
            chosen = torch.full(
                (count,), INVALID_TOKEN_ID, dtype=torch.long, device=device
            )
            route_here = torch.full(
                (count,), ROUTE_PADDING, dtype=torch.int8, device=device
            )
            k_here = torch.zeros(count, dtype=torch.int16, device=device)
            signal_here = torch.full(
                (count,), float("nan"), dtype=torch.float32, device=device
            )

            fast = torch.zeros(count, dtype=torch.bool, device=device)
            if policy.spec.fast_path_threshold is not None:
                fast = eligible & confidence.ge(policy.spec.fast_path_threshold)
                if fast.any().item():
                    chosen[fast] = ids[fast, 0]
                    route_here[fast] = ROUTE_FAST_A1

            pending = torch.nonzero(eligible & ~fast, as_tuple=False).flatten()
            history: torch.Tensor | None = None
            previous = 0
            for stage_index, k in enumerate(policy.spec.schedule):
                if not pending.numel():
                    break
                segment = _candidate_hidden(
                    precut,
                    cache=cache,
                    parent_indices=pending,
                    candidate_ids=ids[pending, previous:k],
                    position=position,
                )
                history = segment if history is None else torch.cat((history, segment), dim=1)
                logical_simulations += int(pending.numel()) * (k - previous)
                executed_simulations += int(pending.numel()) * (k - previous)
                scores = score_candidates(
                    history,
                    target[pending],
                    policy.spec.score_rule,
                )
                winner = scores.argmax(dim=1)
                winner_token = ids[pending, :k].gather(1, winner[:, None]).squeeze(1)
                last = stage_index == len(policy.spec.schedule) - 1
                if policy.spec.kind == "fixed":
                    signal_value = torch.full(
                        (pending.numel(),), float("nan"), device=device
                    )
                    accept = torch.ones(pending.numel(), dtype=torch.bool, device=device)
                else:
                    if policy.spec.routing_signal is None:
                        raise ConfigurationSearchError("adaptive policy lost its signal")
                    signal_value = routing_signal(scores, policy.spec.routing_signal)
                    pass_gate = gate_passes(
                        signal_value,
                        policy.threshold_at(k),
                        policy.spec.gate_comparator,
                    )
                    accept = (
                        torch.ones_like(pass_gate)
                        if last and policy.spec.terminal_action == "commit_last_winner"
                        else pass_gate
                    )
                accepted_offsets = torch.nonzero(accept, as_tuple=False).flatten()
                if accepted_offsets.numel():
                    accepted = pending[accepted_offsets]
                    chosen[accepted] = winner_token[accepted_offsets]
                    route_here[accepted] = ROUTE_TIER
                    k_here[accepted] = k
                    signal_here[accepted] = signal_value[accepted_offsets]
                remaining_offsets = torch.nonzero(~accept, as_tuple=False).flatten()
                pending = pending[remaining_offsets]
                history = history[remaining_offsets]
                if last and pending.numel():
                    if policy.spec.terminal_action == "fallback_to_a1":
                        chosen[pending] = ids[pending, 0]
                        route_here[pending] = ROUTE_FINAL_A1_FALLBACK
                        k_here[pending] = k
                        signal_here[pending] = signal_value[remaining_offsets]
                        pending = pending[:0]
                        history = history[:0]
                    elif policy.spec.terminal_action == "abstain_and_stop_suffix":
                        pending_cpu = pending.cpu()
                        stopped[pending_cpu] = True
                        route_here[pending] = ROUTE_ABSTAIN
                        k_here[pending] = k
                        signal_here[pending] = signal_value[remaining_offsets]
                        pending = pending[:0]
                        history = history[:0]
                    else:
                        raise ConfigurationSearchError("unhandled terminal action")
                previous = k

            recovered = eligible & chosen.ge(0)
            abstained = eligible & ~recovered
            if abstained.any().item() and not stopped[abstained.cpu()].all().item():
                raise ConfigurationSearchError("eligible row neither recovered nor stopped")
            commit[recovered] = chosen[recovered]
            prediction_view = predictions[record_start:record_end, position]
            prediction_view[recovered.cpu()] = chosen[recovered].cpu()
            route_view = routes[record_start:record_end, position]
            route_view[eligible_cpu] = route_here[eligible].cpu()
            selected_k[record_start:record_end, position][eligible_cpu] = k_here[eligible].cpu()
            selected_signal[record_start:record_end, position][eligible_cpu] = (
                signal_here[eligible].cpu()
            )
            precut.run_cached(commit[:, None], cache, position)
            prefix_commit_tokens += count
            del target, chosen, route_here, k_here, signal_here
        del cache

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return DecodeResult(
        predictions=predictions,
        routes=routes,
        selected_k=selected_k,
        selected_signal=selected_signal,
        elapsed_seconds=elapsed,
        candidate_simulations=logical_simulations,
        executed_candidate_simulations=executed_simulations,
        prefix_commit_tokens=prefix_commit_tokens,
        record_batch_size=record_batch_size,
    )


def validate_full_enumeration(specs: Iterable[PolicySpec]) -> dict[str, int]:
    count = 0
    identifiers: set[str] = set()
    current = 0
    historical = 0
    for spec in specs:
        spec.validate()
        count += 1
        if spec.policy_id in identifiers:
            raise ConfigurationSearchError("duplicate serialized policy")
        identifiers.add(spec.policy_id)
        current += int(spec == current_anchor_spec())
        historical += int(spec == historical_anchor_spec())
    if count != declared_policy_count() or len(identifiers) != count:
        raise ConfigurationSearchError("full policy enumeration count changed")
    if current != 1 or historical != 1:
        raise ConfigurationSearchError("anchor multiplicity changed")
    return {
        "declared": declared_policy_count(),
        "enumerated": count,
        "unique": len(identifiers),
        "current_anchor": current,
        "historical_anchor": historical,
    }


__all__ = [
    "ADAPTIVE_GRID",
    "BUDGET_GRID",
    "CURRENT_THRESHOLD",
    "DecodeResult",
    "EMPIRICAL_GATE_MODES",
    "FAST_PATHS",
    "HISTORICAL_GATE_MODE",
    "PolicySpec",
    "ResolvedPolicy",
    "ROUTING_SIGNALS",
    "SCORE_RULES",
    "TERMINAL_ACTIONS",
    "adaptive_schedules",
    "current_anchor_spec",
    "declared_policy_count",
    "decode_policy",
    "gate_passes",
    "historical_anchor_spec",
    "iter_policy_specs",
    "policy_spec_from_dict",
    "resolve_policy",
    "resolved_policy_from_dict",
    "routing_signal",
    "score_candidates",
    "validate_full_enumeration",
]
