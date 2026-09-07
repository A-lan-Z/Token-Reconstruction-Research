"""Production caller plumbing for the P09 fixed-readout control.

This module deliberately contains no training loop, model/data loader, public
forward, or truth access.  It binds public validation metadata to domain views,
reproduces the inherited CPU ``torch.Generator`` position schedule, and turns
the shared runner's in-memory result into create-only fixed-state and run
receipts.  Scientific constants remain caller inputs except for the signed
P09 checkpoint-grid formula and 512-position schedule budget.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any
from types import MappingProxyType

import torch
from safetensors.torch import save as save_safetensors
from torch import nn

from .fixed_control_runner import (
    BatchLike,
    DOMAIN_BALANCED_SELECTION_METRIC,
    FixedControlRunnerError,
    SchedulePlan,
    ScheduleStep,
    aggregate_domain_validation,
    build_run_receipt,
    canonical_digest,
    method_state_digest,
    write_create_only_json,
)
from token_reconstruction.trr_p09_fixed_control_adapter import BankContract, FixedPublicReadoutHook


CALLER_SCHEMA = "token-reconstruction.trr-p09-fixed-control-caller.v1"
FIXED_STATE_SCHEMA = "token-reconstruction.trr-p09-fixed-state.v1"
SIGNED_POSITION_BUDGET = 512
SIGNED_CHECKPOINT_MILESTONES = (0, 1000, 2000, 4000, 8000, 12000)
DEFAULT_VALIDATION_DOMAINS = ("Finance", "Pile")
_SHA256_LENGTH = 64


class FixedControlCallerError(FixedControlRunnerError):
    """Raised when fixed-control caller metadata or output bindings are invalid."""


@dataclass(frozen=True)
class PublicValidationLabel:
    """One public validation row's domain label and global bank identity."""

    global_row: int
    record_id: str
    domain: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_row": int(self.global_row),
            "record_id": self.record_id,
            "domain": self.domain,
        }


@dataclass(frozen=True)
class PublicValidationLabelJoin:
    """Immutable, hash-bound domain partition for public validation batches."""

    required_domains: tuple[str, ...]
    rows: tuple[PublicValidationLabel, ...]
    rows_by_domain: Mapping[str, tuple[int, ...]]
    semantic_sha256: str

    def validate(self) -> None:
        domains = tuple(self.required_domains)
        if not domains or len(set(domains)) != len(domains):
            raise FixedControlCallerError("validation domains must be unique and nonempty")
        if not self.rows:
            raise FixedControlCallerError("validation label join is empty")
        if set(self.rows_by_domain) != set(domains):
            raise FixedControlCallerError("validation domain partition is incomplete")
        seen_rows: set[int] = set()
        seen_ids: set[str] = set()
        expected_by_domain: dict[str, list[int]] = {domain: [] for domain in domains}
        for row in self.rows:
            if isinstance(row.global_row, bool) or not isinstance(row.global_row, int) or row.global_row < 0:
                raise FixedControlCallerError("validation global row is invalid")
            if not row.record_id or not isinstance(row.record_id, str):
                raise FixedControlCallerError("validation record ID is invalid")
            if row.domain not in expected_by_domain:
                raise FixedControlCallerError(f"validation row has unknown domain {row.domain!r}")
            if row.global_row in seen_rows:
                raise FixedControlCallerError("validation global rows are duplicated")
            if row.record_id in seen_ids:
                raise FixedControlCallerError("validation record IDs are duplicated")
            seen_rows.add(row.global_row)
            seen_ids.add(row.record_id)
            expected_by_domain[row.domain].append(row.global_row)
        for domain in domains:
            actual = tuple(int(value) for value in self.rows_by_domain[domain])
            if actual != tuple(expected_by_domain[domain]):
                raise FixedControlCallerError(f"validation row partition changed for {domain!r}")
            if not actual:
                raise FixedControlCallerError(f"validation domain {domain!r} has no rows")
        expected_digest = canonical_digest(
            {
                "required_domains": list(domains),
                "rows": [row.as_dict() for row in self.rows],
                "rows_by_domain": {domain: list(self.rows_by_domain[domain]) for domain in domains},
            }
        )
        if self.semantic_sha256 != expected_digest:
            raise FixedControlCallerError("validation label-join digest changed")
        if len(self.semantic_sha256) != _SHA256_LENGTH:
            raise FixedControlCallerError("validation label-join digest is malformed")

    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "token-reconstruction.trr-p09-public-validation-label-join.v1",
            "required_domains": list(self.required_domains),
            "record_count": len(self.rows),
            "rows_by_domain": {domain: list(self.rows_by_domain[domain]) for domain in self.required_domains},
            "semantic_sha256": self.semantic_sha256,
        }


def join_public_validation_labels(
    records: Iterable[Mapping[str, Any]],
    *,
    required_domains: Sequence[str] = DEFAULT_VALIDATION_DOMAINS,
) -> PublicValidationLabelJoin:
    """Bind public validation domain labels without opening source or truth.

    ``records`` is metadata only.  It must contain ``global_row``, ``record_id``
    and ``domain`` fields.  Token labels remain in the public validation batch
    source passed to the runner; this function only makes the domain join
    explicit and reproducible.
    """

    domains = tuple(str(domain) for domain in required_domains)
    if not domains or len(set(domains)) != len(domains):
        raise FixedControlCallerError("required validation domains must be unique and nonempty")
    rows: list[PublicValidationLabel] = []
    by_domain: dict[str, list[int]] = {domain: [] for domain in domains}
    for record in records:
        if not isinstance(record, Mapping):
            raise FixedControlCallerError("validation metadata rows must be mappings")
        try:
            global_row = record["global_row"]
            record_id = record["record_id"]
            domain = record["domain"]
        except KeyError as exc:
            raise FixedControlCallerError(f"validation metadata lacks {exc.args[0]!r}") from exc
        row = PublicValidationLabel(
            global_row=global_row,
            record_id=record_id,
            domain=domain,
        )
        rows.append(row)
        if row.domain in by_domain:
            by_domain[row.domain].append(row.global_row)
    payload = {
        "required_domains": list(domains),
        "rows": [row.as_dict() for row in rows],
        "rows_by_domain": {domain: list(by_domain[domain]) for domain in domains},
    }
    join = PublicValidationLabelJoin(
        required_domains=domains,
        rows=tuple(rows),
        rows_by_domain=MappingProxyType(
            {domain: tuple(values) for domain, values in by_domain.items()}
        ),
        semantic_sha256=canonical_digest(payload),
    )
    join.validate()
    return join


ValidationBatchFactory = Callable[[int, str, tuple[int, ...]], Iterable[BatchLike]]


def make_domain_validation_callback(
    label_join: PublicValidationLabelJoin,
    batch_factory: ValidationBatchFactory,
) -> Callable[[int, Callable[[Iterable[BatchLike]], Mapping[str, Any]]], Mapping[str, Any]]:
    """Create the runner callback that evaluates each joined public domain.

    The factory receives ``(checkpoint_step, domain, global_rows)`` and must
    return only that frozen domain's public batches.  It cannot alter the
    selection metric: the callback computes the equal-domain aggregate via the
    runner's canonical function and carries the label-join digest as metadata.
    """

    label_join.validate()
    if not callable(batch_factory):
        raise FixedControlCallerError("validation batch factory must be callable")

    def callback(
        step: int,
        evaluate: Callable[[Iterable[BatchLike]], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise FixedControlCallerError("validation checkpoint step is invalid")
        per_domain: dict[str, Mapping[str, Any]] = {}
        for domain in label_join.required_domains:
            rows = tuple(label_join.rows_by_domain[domain])
            batches = batch_factory(step, domain, rows)
            if batches is None:
                raise FixedControlCallerError(f"validation batch factory returned None for {domain!r}")
            per_domain[domain] = evaluate(batches)
        aggregate = aggregate_domain_validation(
            per_domain,
            required_domains=label_join.required_domains,
        )
        return {
            **aggregate,
            "selection_metric": DOMAIN_BALANCED_SELECTION_METRIC,
            "label_join_sha256": label_join.semantic_sha256,
        }

    return callback


def signed_p09_checkpoint_grid(
    actual_valid_positions: int,
    *,
    position_budget: int = SIGNED_POSITION_BUDGET,
) -> tuple[int, ...]:
    """Derive the signed P09 checkpoint grid from the bound bank count.

    The signed formula is ``N=max(12000,1000*ceil(5*positions/(512*1000)))``
    with milestones ``0,1000,2000,4000,8000,12000,N``.  This helper does not
    inspect the bank and therefore cannot select or alter records.
    """

    if isinstance(actual_valid_positions, bool) or not isinstance(actual_valid_positions, int):
        raise FixedControlCallerError("actual valid-position count must be an integer")
    if actual_valid_positions <= 0:
        raise FixedControlCallerError("actual valid-position count must be positive")
    if position_budget != SIGNED_POSITION_BUDGET:
        raise FixedControlCallerError("signed P09 grid requires a 512-position budget")
    denominator = SIGNED_POSITION_BUDGET * 1000
    required = 1000 * ((5 * actual_valid_positions + denominator - 1) // denominator)
    steps = max(12000, required)
    return tuple(sorted(set(SIGNED_CHECKPOINT_MILESTONES + (steps,))))


def inherited_schedule_steps(
    valid_mask: torch.Tensor,
    global_rows: Sequence[int],
    *,
    steps: int,
    seed: int,
    record_batch_size: int = 8,
    position_budget: int = SIGNED_POSITION_BUDGET,
) -> Iterable[ScheduleStep]:
    """Yield the inherited deterministic position schedule as runner steps.

    The CPU ``torch.Generator``/``randperm``/``randint`` ordering is the same
    as the registered public schedule helper.  The generator is lazy, so a
    production caller may consume a serialized/bound schedule instead of
    retaining all steps in Python memory.  ``SchedulePlan`` remains available
    for bounded fixture construction and digest tests.
    """

    if not isinstance(valid_mask, torch.Tensor) or valid_mask.ndim != 2:
        raise FixedControlCallerError("valid mask must be a rank-two tensor")
    if valid_mask.dtype not in (torch.bool, torch.uint8):
        raise FixedControlCallerError("valid mask must be boolean")
    if steps <= 0 or record_batch_size <= 0 or position_budget <= 0:
        raise FixedControlCallerError("schedule dimensions must be positive")
    if len(global_rows) != int(valid_mask.shape[0]) or not global_rows:
        raise FixedControlCallerError("global rows do not match schedule mask")
    rows = tuple(int(row) for row in global_rows)
    if any(row < 0 for row in rows) or len(set(rows)) != len(rows):
        raise FixedControlCallerError("schedule global rows must be unique and nonnegative")
    mask = valid_mask.detach().to(device="cpu", dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    record_count = len(rows)
    for step_index in range(int(steps)):
        if record_count >= record_batch_size:
            selected = torch.randperm(record_count, generator=generator)[:record_batch_size]
        else:
            selected = torch.randint(
                record_count,
                (record_batch_size,),
                generator=generator,
                dtype=torch.long,
            )
        eligible = torch.nonzero(mask.index_select(0, selected), as_tuple=False)
        eligible = eligible[eligible[:, 1] > 0]
        eligible_count = int(eligible.shape[0])
        if eligible_count <= 0:
            raise FixedControlCallerError("schedule batch has no post-BOS positions")
        if eligible_count >= position_budget:
            chosen = torch.randperm(eligible_count, generator=generator)[:position_budget]
            used_replacement = False
        else:
            chosen = torch.randint(
                eligible_count,
                (position_budget,),
                generator=generator,
                dtype=torch.long,
            )
            used_replacement = True
        draws = eligible.index_select(0, chosen)
        yield ScheduleStep(
            step=step_index,
            batch_global_rows=tuple(rows[int(index)] for index in selected.tolist()),
            draw_record_slots=tuple(int(value) for value in draws[:, 0].tolist()),
            draw_position_slots=tuple(int(value) for value in draws[:, 1].tolist()),
            used_replacement=used_replacement,
        )


def materialize_schedule_plan(
    valid_mask: torch.Tensor,
    global_rows: Sequence[int],
    *,
    steps: int,
    seed: int,
    record_batch_size: int = 8,
    position_budget: int = SIGNED_POSITION_BUDGET,
) -> SchedulePlan:
    """Build and validate a bounded schedule plan for a receipt or fixture."""

    plan = SchedulePlan.from_steps(
        seed=seed,
        steps=tuple(
            inherited_schedule_steps(
                valid_mask,
                global_rows,
                steps=steps,
                seed=seed,
                record_batch_size=record_batch_size,
                position_budget=position_budget,
            )
        ),
    )
    plan.validate(
        record_batch_size=record_batch_size,
        position_budget=position_budget,
        sequence_tokens=int(valid_mask.shape[1]),
    )
    return plan


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise FixedControlCallerError(f"{label} must be a lowercase SHA-256")
    return value


def _state_file_record(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FixedControlCallerError(f"{label} is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"label": label, "path": str(path), "bytes": int(path.stat().st_size), "sha256": digest.hexdigest()}


def make_fixed_checkpoint_callback(
    *,
    output_root: Path,
    bank_contract: BankContract,
    bank_manifest_sha256: str,
    base_state_sha256: str,
    fit_manifest_sha256: str,
    method_id: str = "continued_fixed_readout",
) -> Callable[[Mapping[str, Any], nn.Module, FixedPublicReadoutHook], Mapping[str, Any]]:
    """Create a create-only fixed decoder-state checkpoint callback.

    The callback is compatible with ``run_training``.  It serializes only the
    fixed decoder state and external provenance; optimizer state remains a
    separate artifact.  It verifies the live state digest before and after
    serialization and binds the corrected bank SHA supplied by the caller.
    """

    bank_contract.validate()
    bank_manifest_sha256 = _require_sha(bank_manifest_sha256, label="bank manifest SHA-256")
    base_state_sha256 = _require_sha(base_state_sha256, label="base state SHA-256")
    fit_manifest_sha256 = _require_sha(fit_manifest_sha256, label="fit manifest SHA-256")
    if not method_id:
        raise FixedControlCallerError("fixed method ID is required")
    root = Path(output_root)

    def callback(
        point: Mapping[str, Any],
        decoder: nn.Module,
        hook: FixedPublicReadoutHook,
    ) -> Mapping[str, Any]:
        if not isinstance(hook, FixedPublicReadoutHook):
            raise FixedControlCallerError("fixed checkpoint callback received a non-fixed hook")
        if hook.method_id != method_id:
            raise FixedControlCallerError("fixed hook method ID differs from callback binding")
        if any(parameter.requires_grad for parameter in hook.__dict__.values() if isinstance(parameter, nn.Parameter)):
            raise FixedControlCallerError("fixed hook unexpectedly owns trainable parameters")
        try:
            step = int(point["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FixedControlCallerError("checkpoint point lacks an integer step") from exc
        if step < 0:
            raise FixedControlCallerError("checkpoint step cannot be negative")
        state_sha = method_state_digest(decoder, hook)
        state = {
            str(name): value.detach().cpu().contiguous()
            for name, value in decoder.state_dict().items()
        }
        metadata = {
            "schema": FIXED_STATE_SCHEMA,
            "method_id": method_id,
            "selected_step": str(step),
            "bank_manifest_sha256": bank_manifest_sha256,
            "fit_manifest_sha256": fit_manifest_sha256,
            "base_state_sha256": base_state_sha256,
            "schedule_semantic_sha256": bank_contract.schedule_semantic_sha256,
            "embedding_sha256": bank_contract.embedding.sha256,
            "runner_state_sha256": state_sha,
            "serialization_only": "true",
            "optimizer_state_external": "true",
        }
        encoded = save_safetensors(state, metadata=metadata)
        path = root / f"checkpoint_step_{step:06d}.safetensors"
        if path.exists():
            raise FixedControlCallerError(f"checkpoint destination already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        after_sha = method_state_digest(decoder, hook)
        if after_sha != state_sha:
            raise FixedControlCallerError("fixed checkpoint serialization mutated model state")
        record = _state_file_record(path, label="fixed decoder checkpoint")
        return {
            "checkpoint": record,
            "state_sha256": state_sha,
            "serialization_only": True,
            "optimizer_state_external": True,
            "resume_requires_separate_optimizer_artifact": True,
            "bank_manifest_sha256": bank_manifest_sha256,
        }

    return callback


def fixed_control_cost_summary(training_result: Mapping[str, Any]) -> dict[str, Any]:
    """Derive explicit fit/validation/exposure cost fields from runner output."""

    try:
        schedule = training_result["schedule"]
        timing = training_result["timing"]
        curve = training_result["learning_curve"]
        status = training_result["status"]
    except KeyError as exc:
        raise FixedControlCallerError(f"training result lacks {exc.args[0]!r}") from exc
    if status != "COMPLETED" or not isinstance(schedule, Mapping) or not isinstance(timing, Mapping) or not isinstance(curve, Sequence):
        raise FixedControlCallerError("training result is not a completed runner result")
    steps = int(schedule.get("steps", -1))
    exposure = schedule.get("exposure")
    if steps <= 0 or not isinstance(exposure, Mapping):
        raise FixedControlCallerError("training schedule/exposure is incomplete")
    draws_per_step = int(exposure.get("draws_per_step", -1))
    total_draws = int(exposure.get("total_draws", -1))
    if draws_per_step <= 0 or total_draws != steps * draws_per_step:
        raise FixedControlCallerError("training exposure totals are inconsistent")
    finite_timing: dict[str, float] = {}
    for key in (
        "whole_wall_seconds",
        "stream_load_seconds",
        "optimizer_update_seconds",
        "validation_seconds",
    ):
        value = timing.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise FixedControlCallerError(f"training timing field {key!r} is invalid")
        finite_timing[key] = float(value)
    validation_rows = 0
    for point in curve:
        if not isinstance(point, Mapping):
            raise FixedControlCallerError("learning curve point is not a mapping")
        metrics = point.get("validation")
        if isinstance(metrics, Mapping):
            rows = metrics.get("token_rows")
            if isinstance(rows, int) and not isinstance(rows, bool):
                validation_rows += rows
    return {
        "scope": "fixed_control_runner_process",
        "updates": steps,
        "position_draws_per_update": draws_per_step,
        "total_position_draws": total_draws,
        "validation_checkpoint_count": len(curve),
        "validation_token_rows_across_checkpoints": validation_rows,
        "timing": finite_timing,
        "timing_boundary": timing.get("timing_boundary"),
        "optimizer_steps": steps,
        "pretraining_or_ancestor_cost": "not included",
    }


def build_fixed_control_receipt(
    *,
    training_result: Mapping[str, Any],
    source_commit: str,
    command: Sequence[str],
    started_utc: str,
    finished_utc: str,
    environment: Mapping[str, Any],
    assets: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    resource_peak: Mapping[str, Any],
    bank_manifest_sha256: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a complete fixed-control receipt without reopening any input."""

    bank_manifest_sha256 = _require_sha(bank_manifest_sha256, label="bank manifest SHA-256")
    cost = fixed_control_cost_summary(training_result)
    schedule = training_result.get("schedule")
    if not isinstance(schedule, Mapping):
        raise FixedControlCallerError("training result schedule is missing")
    receipt = build_run_receipt(
        source_commit=source_commit,
        command=command,
        started_utc=started_utc,
        finished_utc=finished_utc,
        environment=environment,
        assets={**dict(assets), "bank_manifest_sha256": bank_manifest_sha256},
        training_contract={**dict(training_contract), "selection_metric": DOMAIN_BALANCED_SELECTION_METRIC},
        timing=dict(training_result["timing"]),
        resource_peak=resource_peak,
        exposure=dict(schedule["exposure"]),
        state={
            **dict(state),
            "selected_step": int(training_result["selected_step"]),
            "selected_state_sha256": str(training_result["selected_state_sha256"]),
            "bank_manifest_sha256": bank_manifest_sha256,
            "checkpoint_state_bindings": list(training_result.get("checkpoint_state_bindings", [])),
        },
        learning_curve=list(training_result["learning_curve"]),
        status=str(training_result["status"]),
    )
    receipt["caller_schema"] = CALLER_SCHEMA
    receipt["cost"] = cost
    receipt["selection"] = {
        "metric": DOMAIN_BALANCED_SELECTION_METRIC,
        "rule": "earliest strict maximum; step zero eligible",
        "selected_step": int(training_result["selected_step"]),
    }
    return receipt


def write_fixed_control_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Write a fixed-control receipt once, preserving failed destinations."""

    if receipt.get("caller_schema") != CALLER_SCHEMA:
        raise FixedControlCallerError("receipt was not built by the fixed-control caller")
    try:
        write_create_only_json(Path(path), receipt)
    except FixedControlRunnerError as exc:
        raise FixedControlCallerError(str(exc)) from exc


__all__ = [
    "CALLER_SCHEMA",
    "DEFAULT_VALIDATION_DOMAINS",
    "FixedControlCallerError",
    "FIXED_STATE_SCHEMA",
    "PublicValidationLabel",
    "PublicValidationLabelJoin",
    "build_fixed_control_receipt",
    "fixed_control_cost_summary",
    "inherited_schedule_steps",
    "join_public_validation_labels",
    "make_domain_validation_callback",
    "make_fixed_checkpoint_callback",
    "materialize_schedule_plan",
    "signed_p09_checkpoint_grid",
    "write_fixed_control_receipt",
]
