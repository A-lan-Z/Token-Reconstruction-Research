"""Shared streamed training/validation primitives for TRR-P09.

The runner keeps the public decoder, sampler, validation, and checkpoint
selection outside the method-specific readout hook.  It consumes a lazy batch
source exposing ``batch_for_global_rows``; the P09 bank loader can be wrapped
by :class:`SequentialLoaderSource` when a serialized schedule is ordered by
stream position.  No bank, model checkpoint, source row, or truth asset is
loaded at import time.

The functions are contract-level infrastructure.  Scientific update counts,
validation panel identity, learning rates, and directional loss terms are
supplied by the frozen caller rather than defaulted here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from token_reconstruction.trr_p09_fixed_control_adapter import (
    FixedControlContractError,
    ReadoutHook,
    shared_decoder_rows,
)


RUNNER_SCHEMA = "token-reconstruction.trr-p09-fixed-control-run.v1"


class FixedControlRunnerError(RuntimeError):
    """Raised when a streamed runner contract or batch is invalid."""


class BatchLike(Protocol):
    activations: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    global_rows: tuple[int, ...]


class BatchSource(Protocol):
    """Lazy source for one complete batch identified by global row indices."""

    def batch_for_global_rows(self, global_rows: Sequence[int]) -> BatchLike:
        """Return one batch without materializing the bank."""


class LoaderLike(Protocol):
    def iter_batches(self) -> Iterable[BatchLike]:
        """Yield complete loader batches in immutable stream order."""


ValidationViewEvaluator = Callable[[Iterable[BatchLike]], Mapping[str, Any]]
ValidationCallback = Callable[[int, ValidationViewEvaluator], Mapping[str, Any]]
REQUIRED_VALIDATION_DOMAINS = ("Finance", "Pile")
DOMAIN_BALANCED_SELECTION_METRIC = "domain_balanced_token_accuracy"


class SequentialLoaderSource:
    """Adapt the setup loader when schedule order is the loader stream order.

    This adapter deliberately keeps no cache.  A schedule whose next batch is
    not the next immutable shard batch fails closed; use
    :class:`RandomAccessLoaderSource` for the real arbitrary schedule path.
    """

    def __init__(self, loader: LoaderLike) -> None:
        self._batches = iter(loader.iter_batches())

    def batch_for_global_rows(self, global_rows: Sequence[int]) -> BatchLike:
        expected = tuple(int(row) for row in global_rows)
        try:
            batch = next(self._batches)
        except StopIteration as exc:
            raise FixedControlRunnerError(
                "streamed loader ended before the scheduled batch"
            ) from exc
        actual = tuple(int(row) for row in batch.global_rows)
        if actual != expected:
            raise FixedControlRunnerError(
                f"stream order differs from schedule: expected {expected}, got {actual}"
            )
        return batch


class RandomAccessLoaderSource:
    """Adapt setup's hash-bound arbitrary-row shard loader.

    ``get_records`` is required.  The adapter never falls back to
    ``iter_batches`` because doing so would silently replace the frozen random
    schedule with shard order.  The setup loader preserves cross-shard order
    and repeated global rows; this adapter checks that promise at the boundary.
    """

    def __init__(self, loader: Any) -> None:
        getter = getattr(loader, "get_records", None)
        if not callable(getter):
            raise FixedControlRunnerError(
                "random-access source must provide get_records(global_indices)"
            )
        self._loader = loader
        self._get_records = getter

    def batch_for_global_rows(self, global_rows: Sequence[int]) -> BatchLike:
        expected = tuple(int(row) for row in global_rows)
        if not expected or any(row < 0 for row in expected):
            raise FixedControlRunnerError("scheduled global rows must be nonnegative and nonempty")
        try:
            batch = self._get_records(expected)
        except Exception as exc:
            raise FixedControlRunnerError("random-access loader rejected scheduled rows") from exc
        actual = tuple(int(row) for row in batch.global_rows)
        if actual != expected:
            raise FixedControlRunnerError(
                f"random-access loader changed row order: expected {expected}, got {actual}"
            )
        return batch


@dataclass(frozen=True)
class ScheduleStep:
    """One serialized training update's batch and position draws."""

    step: int
    batch_global_rows: tuple[int, ...]
    draw_record_slots: tuple[int, ...]
    draw_position_slots: tuple[int, ...]
    used_replacement: bool

    def validate(self, *, record_batch_size: int, position_budget: int, sequence_tokens: int) -> None:
        if self.step < 0:
            raise FixedControlRunnerError("schedule step must be nonnegative")
        if len(self.batch_global_rows) != record_batch_size:
            raise FixedControlRunnerError("schedule batch size differs")
        if len(self.draw_record_slots) != position_budget or len(self.draw_position_slots) != position_budget:
            raise FixedControlRunnerError("schedule position budget differs")
        if any(row < 0 for row in self.batch_global_rows):
            raise FixedControlRunnerError("schedule batch rows must be nonnegative")
        if len(set(self.batch_global_rows)) != len(self.batch_global_rows):
            raise FixedControlRunnerError("schedule batch rows must be unique")
        if any(slot < 0 or slot >= record_batch_size for slot in self.draw_record_slots):
            raise FixedControlRunnerError("schedule record slot is outside its batch")
        if any(position <= 0 or position >= sequence_tokens for position in self.draw_position_slots):
            raise FixedControlRunnerError("schedule draws BOS or an invalid position")

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "batch_global_rows": list(self.batch_global_rows),
            "draw_record_slots": list(self.draw_record_slots),
            "draw_position_slots": list(self.draw_position_slots),
            "used_replacement": self.used_replacement,
        }


@dataclass(frozen=True)
class SchedulePlan:
    """Immutable schedule metadata shared by every method on one bank."""

    seed: int
    steps: tuple[ScheduleStep, ...]
    semantic_sha256: str

    @classmethod
    def from_steps(cls, *, seed: int, steps: Sequence[ScheduleStep]) -> "SchedulePlan":
        ordered = tuple(steps)
        payload = {"seed": int(seed), "steps": [step.as_dict() for step in ordered]}
        return cls(seed=int(seed), steps=ordered, semantic_sha256=canonical_digest(payload))

    def validate(self, *, record_batch_size: int, position_budget: int, sequence_tokens: int) -> None:
        if not self.steps:
            raise FixedControlRunnerError("schedule must contain at least one step")
        if tuple(step.step for step in self.steps) != tuple(range(len(self.steps))):
            raise FixedControlRunnerError("schedule steps must start at zero and be contiguous")
        for step in self.steps:
            step.validate(
                record_batch_size=record_batch_size,
                position_budget=position_budget,
                sequence_tokens=sequence_tokens,
            )
        expected = SchedulePlan.from_steps(seed=self.seed, steps=self.steps).semantic_sha256
        if expected != self.semantic_sha256:
            raise FixedControlRunnerError("schedule semantic digest changed")

    def exposure_summary(self) -> dict[str, Any]:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for step in self.steps:
            for record_slot, position in zip(
                step.draw_record_slots, step.draw_position_slots, strict=True
            ):
                pair_counts[(step.batch_global_rows[record_slot], position)] += 1
        return {
            "seed": self.seed,
            "steps": len(self.steps),
            "draws_per_step": len(self.steps[0].draw_position_slots),
            "total_draws": sum(len(step.draw_position_slots) for step in self.steps),
            "used_replacement_steps": sum(step.used_replacement for step in self.steps),
            "unique_record_position_pairs": len(pair_counts),
            "repeated_record_position_pairs": sum(value > 1 for value in pair_counts.values()),
            "max_exposures_per_record_position": max(pair_counts.values(), default=0),
            "schedule_semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True)
class RunnerConfig:
    """All numerical choices are explicit and supplied by the study contract."""

    steps: int
    record_batch_size: int
    position_budget: int
    validation_every: int
    selection_metric: str
    seed: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    train_sequence_tokens: int
    hidden_size: int
    expected_activation_dtype: str | None

    def validate(self) -> None:
        if self.steps <= 0 or self.record_batch_size <= 0 or self.position_budget <= 0:
            raise FixedControlRunnerError("training dimensions must be positive")
        if self.validation_every <= 0 or not self.selection_metric:
            raise FixedControlRunnerError("validation selection contract is incomplete")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise FixedControlRunnerError("optimizer settings are invalid")
        if self.train_sequence_tokens <= 1 or self.hidden_size <= 0:
            raise FixedControlRunnerError("training geometry is invalid")


@dataclass(frozen=True)
class StepResult:
    step: int
    loss: float
    cross_entropy_loss: float
    token_rows: int
    correct_tokens: int
    token_accuracy: float
    gradient_norm: float
    elapsed_seconds: float
    load_seconds: float
    update_seconds: float
    batch_global_rows: tuple[int, ...]
    used_replacement: bool
    state_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FixedControlRunnerError("value cannot be canonically encoded") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_bytes({"shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return hashlib.sha256(header + tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")).hexdigest()


def _trainable_parameters(value: Any) -> list[nn.Parameter]:
    parameters = getattr(value, "parameters", None)
    if not callable(parameters):
        return []
    return [parameter for parameter in parameters() if parameter.requires_grad]


def optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    """Return unique optimizer parameters in deterministic group order."""

    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group.get("params", ()):
            if not isinstance(parameter, nn.Parameter):
                raise FixedControlRunnerError("optimizer group contains a non-parameter")
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    if not result:
        raise FixedControlRunnerError("optimizer has no parameters")
    return tuple(result)


def validate_optimizer_coverage(
    decoder: nn.Module,
    hook: ReadoutHook,
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Parameter, ...]:
    """Require decoder and hook trainable parameters to share one optimizer."""

    parameters = optimizer_parameters(optimizer)
    trainable_parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable_parameters:
        raise FixedControlRunnerError("optimizer has no trainable parameters")
    identities = {id(parameter) for parameter in trainable_parameters}
    owned_parameters = tuple(_trainable_parameters(decoder)) + tuple(_trainable_parameters(hook))
    owned_identities = {id(parameter) for parameter in owned_parameters}
    decoder_missing = [
        parameter for parameter in _trainable_parameters(decoder) if id(parameter) not in identities
    ]
    hook_missing = [
        parameter for parameter in _trainable_parameters(hook) if id(parameter) not in identities
    ]
    unowned = [parameter for parameter in trainable_parameters if id(parameter) not in owned_identities]
    if decoder_missing or hook_missing:
        raise FixedControlRunnerError(
            "optimizer omits trainable decoder/readout parameters"
        )
    if unowned:
        raise FixedControlRunnerError(
            "optimizer contains trainable parameters outside decoder/readout contract"
        )
    return trainable_parameters


def _state_entries(value: Any, *, prefix: str) -> list[tuple[str, torch.Tensor]]:
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        return []
    return [(f"{prefix}{name}", tensor) for name, tensor in state_dict().items()]


def method_state_digest(decoder: nn.Module, hook: ReadoutHook) -> str:
    """Hash decoder plus hook state for checkpoint receipts, not every step."""

    digest = hashlib.sha256()
    entries = _state_entries(decoder, prefix="decoder::") + _state_entries(hook, prefix="readout::")
    for name, value in sorted(entries):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical_bytes({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_batch(
    batch: BatchLike,
    *,
    expected_global_rows: Sequence[int] | None,
    expected_sequence_tokens: int,
    expected_hidden_size: int,
    expected_batch_records: int,
    expected_activation_dtype: torch.dtype | None = None,
) -> None:
    activations = batch.activations
    if activations.ndim != 3 or tuple(activations.shape) != (
        expected_batch_records,
        expected_sequence_tokens,
        expected_hidden_size,
    ):
        raise FixedControlRunnerError(
            f"activation geometry differs: {tuple(activations.shape)}"
        )
    if expected_activation_dtype is not None and activations.dtype != expected_activation_dtype:
        raise FixedControlRunnerError(
            f"activation dtype differs: {activations.dtype} != {expected_activation_dtype}"
        )
    expected_shape = (expected_batch_records, expected_sequence_tokens)
    for label, tensor in (
        ("token_ids", batch.token_ids),
        ("attention_mask", batch.attention_mask),
        ("position_ids", batch.position_ids),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise FixedControlRunnerError(f"{label} geometry differs")
    if len(batch.global_rows) != expected_batch_records:
        raise FixedControlRunnerError("batch global row count differs")
    if any(int(row) < 0 for row in batch.global_rows):
        raise FixedControlRunnerError("batch global rows must be nonnegative")
    if expected_global_rows is not None and tuple(batch.global_rows) != tuple(int(row) for row in expected_global_rows):
        raise FixedControlRunnerError("batch global row order differs from schedule")
    mask = batch.attention_mask.to(dtype=torch.bool)
    if not bool(mask[:, 0].all().item()):
        raise FixedControlRunnerError("batch contains a sequence without BOS")
    positions = batch.position_ids.to(dtype=torch.long)
    expected_positions = torch.arange(expected_sequence_tokens, device=positions.device).expand(
        expected_batch_records, -1
    )
    # P09 bank sidecars use absolute positions on the active prefix and zero
    # for right-padding.  Keep this validator aligned with the immutable bank
    # loader; requiring arange in inactive rows would reject valid streamed
    # records before the fixed or directional hook sees them.
    expected_positions = torch.where(mask, expected_positions, torch.zeros_like(expected_positions))
    if not torch.equal(positions, expected_positions):
        raise FixedControlRunnerError("batch position IDs do not match active-prefix/zero-padding semantics")


def _device_for(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise FixedControlRunnerError("decoder has no parameters") from exc


def _move_embedding_once(embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
    if embedding.device != device:
        raise FixedControlRunnerError(
            "embedding must be staged on the decoder device once before the loop"
        )
    if not embedding.is_floating_point() or embedding.ndim != 2:
        raise FixedControlRunnerError("embedding must be a floating rank-two tensor")
    return embedding


def train_one_step(
    decoder: nn.Module,
    hook: ReadoutHook,
    source: BatchSource,
    schedule_step: ScheduleStep,
    *,
    optimizer: torch.optim.Optimizer,
    embedding: torch.Tensor,
    config: RunnerConfig,
    activation_dtype: torch.dtype | None = None,
    compute_base_logits: bool = True,
    record_state_digest: bool = False,
) -> StepResult:
    """Run one scheduled update against one lazy eight-record batch."""

    device = _device_for(decoder)
    _move_embedding_once(embedding, device)
    schedule_step.validate(
        record_batch_size=config.record_batch_size,
        position_budget=config.position_budget,
        sequence_tokens=config.train_sequence_tokens,
    )
    started = time.perf_counter()
    load_started = time.perf_counter()
    batch = source.batch_for_global_rows(schedule_step.batch_global_rows)
    validate_batch(
        batch,
        expected_global_rows=schedule_step.batch_global_rows,
        expected_sequence_tokens=config.train_sequence_tokens,
        expected_hidden_size=config.hidden_size,
        expected_batch_records=config.record_batch_size,
        expected_activation_dtype=activation_dtype,
    )
    load_seconds = time.perf_counter() - load_started
    update_started = time.perf_counter()
    decoder.train()
    if isinstance(hook, nn.Module):
        hook.train()
    activation = batch.activations.to(device=device, dtype=torch.float32)
    mask = batch.attention_mask.to(device=device, dtype=torch.bool)
    token_ids = batch.token_ids.to(device=device, dtype=torch.long)
    record_slots = torch.tensor(schedule_step.draw_record_slots, device=device, dtype=torch.long)
    position_slots = torch.tensor(schedule_step.draw_position_slots, device=device, dtype=torch.long)
    sampled_mask = mask[record_slots, position_slots]
    if not bool(sampled_mask.all().item()):
        raise FixedControlRunnerError("schedule sampled a masked/invalid position")
    target_ids = token_ids[record_slots, position_slots]
    optimizer_parameters_for_step = validate_optimizer_coverage(decoder, hook, optimizer)
    optimizer.zero_grad(set_to_none=True)
    try:
        _, logits, losses = shared_decoder_rows(
            decoder,
            hook,
            activation,
            mask,
            record_slots,
            position_slots,
            embedding,
            target_ids,
            compute_base_logits=compute_base_logits,
        )
    except FixedControlContractError as exc:
        raise FixedControlRunnerError(str(exc)) from exc
    total = losses["total"]
    if not torch.isfinite(total).item():
        raise FixedControlRunnerError("training loss is non-finite")
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        optimizer_parameters_for_step, config.gradient_clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    for parameter in optimizer_parameters_for_step:
        if not torch.isfinite(parameter).all().item():
            raise FixedControlRunnerError("optimizer parameter became non-finite")
    prediction = logits.detach().argmax(dim=-1)
    cross_entropy = losses.get("cross_entropy", total)
    return StepResult(
        step=schedule_step.step + 1,
        loss=float(total.detach().cpu()),
        cross_entropy_loss=float(cross_entropy.detach().cpu()),
        token_rows=int(target_ids.numel()),
        correct_tokens=int(prediction.eq(target_ids).sum().detach().cpu()),
        token_accuracy=float(prediction.eq(target_ids).float().mean().detach().cpu()),
        gradient_norm=float(torch.as_tensor(gradient_norm).detach().cpu()),
        elapsed_seconds=time.perf_counter() - started,
        load_seconds=load_seconds,
        update_seconds=time.perf_counter() - update_started,
        batch_global_rows=tuple(schedule_step.batch_global_rows),
        used_replacement=bool(schedule_step.used_replacement),
        state_sha256=(method_state_digest(decoder, hook) if record_state_digest else None),
    )


def evaluate_batches(
    decoder: nn.Module,
    hook: ReadoutHook,
    batches: Iterable[BatchLike],
    *,
    embedding: torch.Tensor,
    expected_sequence_tokens: int,
    expected_hidden_size: int,
    expected_batch_records: int,
    position_budget: int,
    activation_dtype: torch.dtype | None = None,
    compute_base_logits: bool = True,
) -> dict[str, Any]:
    """Evaluate a validation view whose sequence width may differ from fit."""

    device = _device_for(decoder)
    _move_embedding_once(embedding, device)
    if position_budget <= 0:
        raise FixedControlRunnerError("validation position budget must be positive")
    decoder.eval()
    if isinstance(hook, nn.Module):
        hook.eval()
    total_rows = 0
    correct = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in batches:
            validate_batch(
                batch,
                expected_global_rows=None,
                expected_sequence_tokens=expected_sequence_tokens,
                expected_hidden_size=expected_hidden_size,
                expected_batch_records=expected_batch_records,
                expected_activation_dtype=activation_dtype,
            )
            activation = batch.activations.to(device=device, dtype=torch.float32)
            mask = batch.attention_mask.to(device=device, dtype=torch.bool)
            token_ids = batch.token_ids.to(device=device, dtype=torch.long)
            projected = decoder.projected_hidden(activation, mask)
            indices = torch.nonzero(mask, as_tuple=False)
            indices = indices[indices[:, 1] > 0]
            for chunk in indices.split(position_budget):
                record_slots = chunk[:, 0].to(device=device)
                position_slots = chunk[:, 1].to(device=device)
                base_logits = None
                if compute_base_logits:
                    base_logits = decoder.logits_from_rows(
                        projected, record_slots, position_slots, embedding
                    )
                query_rows = projected[record_slots, position_slots]
                scale = getattr(decoder, "logit_scale", None)
                if not isinstance(scale, torch.Tensor):
                    raise FixedControlRunnerError("common decoder must expose tensor logit_scale")
                logits = hook.score_rows(
                    query_rows,
                    scale,
                    embedding,
                    base_logits=base_logits,
                )
                targets = token_ids[record_slots, position_slots]
                loss_sum += float(
                    F.cross_entropy(logits, targets, reduction="sum").detach().cpu()
                )
                correct += int(logits.argmax(dim=-1).eq(targets).sum().detach().cpu())
                total_rows += int(targets.numel())
    if total_rows <= 0:
        raise FixedControlRunnerError("validation view contains no post-BOS rows")
    return {
        "token_rows": total_rows,
        "correct_tokens": correct,
        "token_accuracy": correct / total_rows,
        "cross_entropy_loss": loss_sum / total_rows,
        "compute_base_logits": compute_base_logits,
    }


def aggregate_domain_validation(
    per_domain: Mapping[str, Mapping[str, Any]],
    *,
    required_domains: Sequence[str] = REQUIRED_VALIDATION_DOMAINS,
) -> dict[str, Any]:
    """Aggregate per-domain ``evaluate_batches`` results.

    The pooled accuracy remains a diagnostic.  Checkpoint selection must use
    the explicitly returned equal-domain score.  Exact domain membership and
    count/accuracy consistency fail closed.
    """

    domains = tuple(str(domain) for domain in required_domains)
    if not domains or len(set(domains)) != len(domains):
        raise FixedControlRunnerError("required validation domains must be unique and nonempty")
    if not isinstance(per_domain, Mapping) or set(per_domain) != set(domains):
        actual = sorted(str(key) for key in per_domain) if isinstance(per_domain, Mapping) else []
        raise FixedControlRunnerError(
            f"validation domains must be exactly {list(domains)!r}; got {actual!r}"
        )

    canonical_domains: dict[str, dict[str, Any]] = {}
    total_rows = 0
    total_correct = 0
    balanced_values: list[float] = []
    weighted_losses: list[tuple[int, float]] = []
    for domain in domains:
        metrics = per_domain.get(domain)
        if not isinstance(metrics, Mapping):
            raise FixedControlRunnerError(f"validation metrics for {domain!r} are missing")
        rows = metrics.get("token_rows")
        correct = metrics.get("correct_tokens")
        accuracy = metrics.get("token_accuracy")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise FixedControlRunnerError(f"validation token_rows for {domain!r} are invalid")
        if isinstance(correct, bool) or not isinstance(correct, int) or correct < 0 or correct > rows:
            raise FixedControlRunnerError(f"validation correct_tokens for {domain!r} are invalid")
        expected_accuracy = correct / rows
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
            raise FixedControlRunnerError(f"validation token_accuracy for {domain!r} is missing")
        if not math.isfinite(float(accuracy)) or not math.isclose(
            float(accuracy), expected_accuracy, rel_tol=0.0, abs_tol=1e-12
        ):
            raise FixedControlRunnerError(f"validation token_accuracy for {domain!r} is inconsistent")
        canonical = dict(metrics)
        canonical["token_rows"] = rows
        canonical["correct_tokens"] = correct
        canonical["token_accuracy"] = expected_accuracy
        canonical_domains[domain] = canonical
        total_rows += rows
        total_correct += correct
        balanced_values.append(expected_accuracy)
        loss = metrics.get("cross_entropy_loss")
        if loss is not None:
            if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
                raise FixedControlRunnerError(f"validation cross_entropy_loss for {domain!r} is invalid")
            weighted_losses.append((rows, float(loss)))

    result: dict[str, Any] = {
        "token_rows": total_rows,
        "correct_tokens": total_correct,
        "token_accuracy": total_correct / total_rows,
        "domain_balanced_token_accuracy": sum(balanced_values) / len(balanced_values),
        "domain_order": list(domains),
        "domains": canonical_domains,
    }
    if len(weighted_losses) == len(domains):
        result["cross_entropy_loss"] = sum(rows * loss for rows, loss in weighted_losses) / total_rows
    return result


def _validate_domain_balanced_metrics(
    metrics: Mapping[str, Any],
    *,
    required_domains: Sequence[str] = REQUIRED_VALIDATION_DOMAINS,
) -> dict[str, Any]:
    """Validate callback output before checkpoint selection."""

    if not isinstance(metrics, Mapping):
        raise FixedControlRunnerError("domain validation callback must return a mapping")
    canonical = aggregate_domain_validation(metrics.get("domains"), required_domains=required_domains)
    supplied = metrics.get(DOMAIN_BALANCED_SELECTION_METRIC)
    if isinstance(supplied, bool) or not isinstance(supplied, (int, float)):
        raise FixedControlRunnerError(
            f"domain validation callback must return {DOMAIN_BALANCED_SELECTION_METRIC!r}"
        )
    if not math.isfinite(float(supplied)) or not math.isclose(
        float(supplied), canonical[DOMAIN_BALANCED_SELECTION_METRIC], rel_tol=0.0, abs_tol=1e-12
    ):
        raise FixedControlRunnerError("domain-balanced selection metric is inconsistent with domain metrics")
    return {**dict(metrics), **canonical}


def run_training(
    decoder: nn.Module,
    hook: ReadoutHook,
    source: BatchSource,
    schedule_steps: Iterable[ScheduleStep],
    *,
    schedule_steps_count: int,
    schedule_seed: int,
    schedule_semantic_sha256: str,
    schedule_exposure: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    embedding: torch.Tensor,
    config: RunnerConfig,
    validation_batches: Callable[[int], Iterable[BatchLike]] | None = None,
    validation_callback: ValidationCallback | None = None,
    validation_sequence_tokens: int,
    validation_batch_records: int,
    validation_activation_dtype: torch.dtype | None,
    training_activation_dtype: torch.dtype | None,
    checkpoint_steps: Sequence[int],
    scheduler: Any | None = None,
    compute_base_logits: bool = True,
    checkpoint_callback: Callable[
        [Mapping[str, Any], nn.Module, ReadoutHook], Mapping[str, Any] | None
    ]
    | None = None,
    deadline_seconds: float | None = None,
    resource_guard_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one fixed/directional continuation under one shared schedule.

    ``schedule_steps`` is an iterable instead of a Python list so a production
    safetensors schedule reader can stream 12,000 steps without materializing
    millions of Python integers. The caller binds the serialized schedule
    digest and exposure summary before calling this function. An explicit
    validation callback may call the supplied per-view evaluator once per
    frozen domain and must return ``aggregate_domain_validation`` output.
    Domain-balanced selection therefore fails closed when the callback or a
    required domain is missing.
    """

    config.validate()
    if schedule_steps_count != config.steps:
        raise FixedControlRunnerError("schedule step count differs from training contract")
    if len(schedule_semantic_sha256) != 64 or schedule_semantic_sha256.lower() != schedule_semantic_sha256:
        raise FixedControlRunnerError("schedule semantic digest is malformed")
    checkpoints = tuple(sorted({int(step) for step in checkpoint_steps}))
    if not checkpoints or checkpoints[0] != 0:
        raise FixedControlRunnerError("checkpoint grid must include step zero")
    if any(step < 0 or step > config.steps for step in checkpoints):
        raise FixedControlRunnerError("checkpoint grid contains an out-of-range step")
    if deadline_seconds is not None and deadline_seconds <= 0:
        raise FixedControlRunnerError("deadline must be positive")
    if resource_guard_callback is not None and not callable(resource_guard_callback):
        raise FixedControlRunnerError("resource guard callback must be callable")
    if config.selection_metric == DOMAIN_BALANCED_SELECTION_METRIC and validation_callback is None:
        raise FixedControlRunnerError(
            "domain-balanced selection requires an explicit validation callback"
        )
    if validation_callback is not None and config.selection_metric != DOMAIN_BALANCED_SELECTION_METRIC:
        raise FixedControlRunnerError(
            "domain validation callback requires domain-balanced selection metric"
        )
    if validation_callback is None and validation_batches is None:
        raise FixedControlRunnerError("validation batches are required without a validation callback")
    validate_optimizer_coverage(decoder, hook, optimizer)
    device = _device_for(decoder)
    _move_embedding_once(embedding, device)
    started = time.perf_counter()
    update_seconds = 0.0
    load_seconds = 0.0
    validation_seconds = 0.0
    points: list[dict[str, Any]] = []
    checkpoint_bindings: list[Mapping[str, Any]] = []
    last_train: StepResult | None = None

    def record_checkpoint(step: int, train_result: StepResult | None) -> None:
        nonlocal validation_seconds
        if resource_guard_callback is not None:
            resource_guard_callback(f"before_validation_step_{step}")
        validation_started = time.perf_counter()
        def evaluate_view(batches: Iterable[BatchLike]) -> Mapping[str, Any]:
            return evaluate_batches(
                decoder,
                hook,
                batches,
                embedding=embedding,
                expected_sequence_tokens=validation_sequence_tokens,
                expected_hidden_size=config.hidden_size,
                expected_batch_records=validation_batch_records,
                position_budget=config.position_budget,
                activation_dtype=validation_activation_dtype,
                compute_base_logits=compute_base_logits,
            )

        if validation_callback is None:
            assert validation_batches is not None
            metrics = dict(evaluate_view(validation_batches(step)))
        else:
            metrics = _validate_domain_balanced_metrics(validation_callback(step, evaluate_view))
        elapsed = time.perf_counter() - validation_started
        validation_seconds += elapsed
        point: dict[str, Any] = {
            "step": int(step),
            "validation": metrics,
            "train": None if train_result is None else train_result.as_dict(),
            "validation_seconds": elapsed,
            "state_sha256": method_state_digest(decoder, hook),
        }
        if checkpoint_callback is not None:
            # Serializers cannot inject or rewrite the metric used for selection.
            callback_point = MappingProxyType(deepcopy(point))
            binding = checkpoint_callback(callback_point, decoder, hook)
            if binding is not None:
                point["state_binding"] = dict(binding)
                checkpoint_bindings.append(dict(binding))
        points.append(point)

    record_checkpoint(0, None)
    schedule_iterator = iter(schedule_steps)
    for step_index in range(config.steps):
        if resource_guard_callback is not None:
            resource_guard_callback(f"before_update_step_{step_index + 1}")
        try:
            schedule_step = next(schedule_iterator)
        except StopIteration as exc:
            raise FixedControlRunnerError("schedule ended before all updates") from exc
        if schedule_step.step != step_index:
            raise FixedControlRunnerError(
                f"schedule step index differs: expected {step_index}, got {schedule_step.step}"
            )
        last_train = train_one_step(
            decoder,
            hook,
            source,
            schedule_step,
            optimizer=optimizer,
            embedding=embedding,
            config=config,
            activation_dtype=training_activation_dtype,
            compute_base_logits=compute_base_logits,
            record_state_digest=False,
        )
        load_seconds += last_train.load_seconds
        update_seconds += last_train.update_seconds
        if scheduler is not None:
            scheduler.step()
        if resource_guard_callback is not None:
            resource_guard_callback(f"after_update_step_{step_index + 1}")
        completed_step = step_index + 1
        if completed_step in checkpoints:
            record_checkpoint(completed_step, last_train)
        if deadline_seconds is not None and time.perf_counter() - started > deadline_seconds:
            raise FixedControlRunnerError("training deadline exceeded")
    try:
        next(schedule_iterator)
    except StopIteration:
        pass
    else:
        raise FixedControlRunnerError("schedule contains more steps than training contract")
    selected = select_earliest_maximum(points, metric=config.selection_metric)
    return {
        "schema": RUNNER_SCHEMA,
        "status": "COMPLETED",
        "schedule": {
            "seed": int(schedule_seed),
            "steps": int(schedule_steps_count),
            "semantic_sha256": schedule_semantic_sha256,
            "exposure": dict(schedule_exposure),
        },
        "checkpoints": list(checkpoints),
        "selected_step": int(selected["step"]),
        "selected_state_sha256": str(selected["state_sha256"]),
        "checkpoint_state_bindings": checkpoint_bindings,
        "learning_curve": points,
        "timing": {
            "whole_wall_seconds": time.perf_counter() - started,
            "stream_load_seconds": load_seconds,
            "optimizer_update_seconds": update_seconds,
            "validation_seconds": validation_seconds,
            "timing_boundary": "includes lazy batch load, decoder update, and validation synchronization as observed by this process",
        },
    }


def _point_metric(point: Mapping[str, Any], metric: str) -> float:
    value: Any = point.get(metric)
    if value is None and isinstance(point.get("validation"), Mapping):
        value = point["validation"].get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixedControlRunnerError(f"checkpoint lacks numeric selection metric {metric!r}")
    score = float(value)
    if not math.isfinite(score):
        raise FixedControlRunnerError(f"checkpoint metric {metric!r} is non-finite")
    return score


def select_earliest_maximum(
    points: Sequence[Mapping[str, Any]], *, metric: str
) -> Mapping[str, Any]:
    """Select the earliest strict maximum, with step zero eligible."""

    if not points:
        raise FixedControlRunnerError("no validation checkpoints were supplied")
    ordered = sorted(points, key=lambda point: int(point["step"]))
    steps = [int(point["step"]) for point in ordered]
    if steps[0] != 0 or len(set(steps)) != len(steps):
        raise FixedControlRunnerError("validation checkpoints must include unique step zero")
    best = ordered[0]
    best_score = _point_metric(best, metric)
    for point in ordered[1:]:
        score = _point_metric(point, metric)
        if score > best_score:
            best = point
            best_score = score
    return best


def build_run_receipt(
    *,
    source_commit: str,
    command: Sequence[str],
    started_utc: str,
    finished_utc: str,
    environment: Mapping[str, Any],
    assets: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    timing: Mapping[str, Any],
    resource_peak: Mapping[str, Any],
    exposure: Mapping[str, Any],
    state: Mapping[str, Any],
    learning_curve: Sequence[Mapping[str, Any]],
    status: str,
) -> dict[str, Any]:
    """Build a complete create-only run receipt without running a fit."""

    if not source_commit or not command or not started_utc or not finished_utc:
        raise FixedControlRunnerError("run provenance is incomplete")
    receipt = {
        "schema": RUNNER_SCHEMA,
        "task_id": "TRR-P09",
        "status": status,
        "source_commit": source_commit,
        "command": [str(arg) for arg in command],
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "environment": dict(environment),
        "assets": dict(assets),
        "training_contract": dict(training_contract),
        "timing": dict(timing),
        "resource_peak": dict(resource_peak),
        "exposure": dict(exposure),
        "state": dict(state),
        "learning_curve": [dict(point) for point in learning_curve],
    }
    canonical_bytes(receipt)
    return receipt


def write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one immutable JSON receipt and refuse an existing destination."""

    path = Path(path)
    if path.exists():
        raise FixedControlRunnerError(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
