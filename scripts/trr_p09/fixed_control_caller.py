"""Production caller plumbing for the P09 fixed-readout control.

This module deliberately contains no training loop, model/data loader, public
forward, or truth access.  It binds public validation metadata to domain views,
reproduces the inherited CPU ``torch.Generator`` position schedule, and turns
the shared runner's in-memory result into create-only fixed-state and run
receipts.  Scientific constants remain caller inputs except for the signed
P09 checkpoint-grid formula and 512-position schedule budget.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any
from types import MappingProxyType

import torch
from safetensors import safe_open
from safetensors.torch import save as save_safetensors
from torch import nn

from .fixed_control_runner import (
    BatchLike,
    BatchSource,
    DOMAIN_BALANCED_SELECTION_METRIC,
    FixedControlRunnerError,
    SchedulePlan,
    ScheduleStep,
    aggregate_domain_validation,
    build_run_receipt,
    evaluate_batches,
    canonical_digest,
    method_state_digest,
    write_create_only_json,
)
from token_reconstruction.trr_p09_fixed_control_adapter import BankContract, FixedPublicReadoutHook, ReadoutHook


CALLER_SCHEMA = "token-reconstruction.trr-p09-fixed-control-caller.v1"
FIXED_STATE_SCHEMA = "token-reconstruction.trr-p09-fixed-state.v1"
SIGNED_POSITION_BUDGET = 512
SIGNED_CHECKPOINT_MILESTONES = (0, 1000, 2000, 4000, 8000, 12000)
DEFAULT_VALIDATION_DOMAINS = ("Finance", "Pile")
_SHA256_LENGTH = 64
COMMON_SCHEDULE_SCHEMA = "token-reconstruction.trr-p09-common-schedules.v1"


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



@dataclass(frozen=True)
class SerializedSchedule:
    """Validated common schedule backed by four CPU tensors.

    The tensors are loaded once after the metadata/hash gate and are retained
    as compact integer arrays. ``iter_steps`` creates one ``ScheduleStep`` at
    a time, so the training loop never materializes the millions of Python
    integers represented by a 13,000-step schedule.
    """

    path: Path
    file_sha256: str
    bank: str
    seed: int
    steps: int
    record_batch_size: int
    position_budget: int
    sequence_tokens: int
    semantic_sha256: str
    valid_mask_semantic_sha256: str
    batch_record_indices: torch.Tensor
    draw_record_slots: torch.Tensor
    draw_position_slots: torch.Tensor
    used_replacement: torch.Tensor

    def iter_steps(self) -> Iterator[ScheduleStep]:
        for index in range(self.steps):
            yield ScheduleStep(
                step=index,
                batch_global_rows=tuple(int(value) for value in self.batch_record_indices[index].tolist()),
                draw_record_slots=tuple(int(value) for value in self.draw_record_slots[index].tolist()),
                draw_position_slots=tuple(int(value) for value in self.draw_position_slots[index].tolist()),
                used_replacement=bool(int(self.used_replacement[index].item())),
            )

    def exposure_summary(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "steps": int(self.steps),
            "draws_per_step": int(self.position_budget),
            "total_draws": int(self.steps * self.position_budget),
            "used_replacement_steps": int(self.used_replacement.sum().item()),
            "schedule_semantic_sha256": self.semantic_sha256,
            "source": "validated_serialized_common_schedule",
        }


def _schedule_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schedule_metadata_int(metadata: Mapping[str, str], key: str) -> int:
    value = metadata.get(key)
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise FixedControlCallerError(f"serialized schedule metadata {key!r} is malformed") from exc
    return parsed


def _schedule_step_bytes(step: ScheduleStep) -> bytes:
    return json.dumps(
        step.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_serialized_schedule(
    path: Path,
    *,
    expected_seed: int,
    expected_steps: int,
    expected_record_batch_size: int,
    expected_position_budget: int,
    expected_sequence_tokens: int,
    expected_global_row_exclusive: int,
    expected_bank: str | None = None,
    expected_valid_mask_semantic_sha256: str | None = None,
    expected_file_sha256: str | None = None,
) -> SerializedSchedule:
    """Load and fully validate one common schedule before model load.

    The B0 artifact contains rows ``[0,1200)`` and the B1 artifact contains
    the composed namespace ``[0,12000)``. Row values are already in the
    namespace consumed by ``CombinedB0StreamedBankLoader``; this function has
    no implicit offset translation.
    """

    schedule_path = Path(path).expanduser().resolve()
    if schedule_path.is_symlink() or not schedule_path.is_file():
        raise FixedControlCallerError(f"serialized schedule is unavailable: {schedule_path}")
    actual_file_sha256 = _schedule_file_sha256(schedule_path)
    if expected_file_sha256 is not None and actual_file_sha256 != expected_file_sha256:
        raise FixedControlCallerError("serialized schedule file hash differs from the supplied binding")

    required_keys = {
        "batch_record_indices",
        "draw_record_slots",
        "draw_position_slots",
        "used_replacement",
    }
    try:
        with safe_open(str(schedule_path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            if metadata.get("schema") != COMMON_SCHEDULE_SCHEMA:
                raise FixedControlCallerError("serialized schedule schema differs")
            if set(handle.keys()) != required_keys:
                raise FixedControlCallerError("serialized schedule tensor keys differ")
            tensors = {
                key: handle.get_tensor(key).contiguous()
                for key in sorted(required_keys)
            }
    except FixedControlCallerError:
        raise
    except Exception as exc:
        raise FixedControlCallerError(f"cannot read serialized schedule: {schedule_path}") from exc

    bank = str(metadata.get("bank", ""))
    if bank not in {"B0", "B1"}:
        raise FixedControlCallerError("serialized schedule bank is invalid")
    if expected_bank is not None and bank != expected_bank:
        raise FixedControlCallerError("serialized schedule bank differs from the requested arm")
    seed = _schedule_metadata_int(metadata, "seed")
    steps = _schedule_metadata_int(metadata, "steps")
    record_batch_size = _schedule_metadata_int(metadata, "record_batch_size")
    position_budget = _schedule_metadata_int(metadata, "position_budget")
    sequence_tokens = _schedule_metadata_int(metadata, "sequence_tokens")
    expected = {
        "seed": int(expected_seed),
        "steps": int(expected_steps),
        "record_batch_size": int(expected_record_batch_size),
        "position_budget": int(expected_position_budget),
        "sequence_tokens": int(expected_sequence_tokens),
    }
    actual = {
        "seed": seed,
        "steps": steps,
        "record_batch_size": record_batch_size,
        "position_budget": position_budget,
        "sequence_tokens": sequence_tokens,
    }
    if actual != expected:
        raise FixedControlCallerError(f"serialized schedule contract differs: expected {expected}, got {actual}")
    if steps <= 0 or expected_global_row_exclusive <= 0:
        raise FixedControlCallerError("serialized schedule dimensions are invalid")

    expected_shapes = {
        "batch_record_indices": (steps, record_batch_size),
        "draw_record_slots": (steps, position_budget),
        "draw_position_slots": (steps, position_budget),
        "used_replacement": (steps,),
    }
    expected_dtypes = {
        "batch_record_indices": torch.int32,
        "draw_record_slots": torch.int16,
        "draw_position_slots": torch.int16,
        "used_replacement": torch.uint8,
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise FixedControlCallerError(f"serialized schedule {key} shape differs")
        if tensors[key].dtype != expected_dtypes[key]:
            raise FixedControlCallerError(f"serialized schedule {key} dtype differs")
    batch_rows = tensors["batch_record_indices"]
    record_slots = tensors["draw_record_slots"]
    positions = tensors["draw_position_slots"]
    replacement = tensors["used_replacement"]
    if bool((batch_rows < 0).any().item()) or bool((batch_rows >= expected_global_row_exclusive).any().item()):
        raise FixedControlCallerError("serialized schedule contains an out-of-range global row")
    if bool((record_slots < 0).any().item()) or bool((record_slots >= record_batch_size).any().item()):
        raise FixedControlCallerError("serialized schedule contains an out-of-range record slot")
    if bool((positions <= 0).any().item()) or bool((positions >= sequence_tokens).any().item()):
        raise FixedControlCallerError("serialized schedule contains BOS or out-of-range position draws")
    if bool(((replacement != 0) & (replacement != 1)).any().item()):
        raise FixedControlCallerError("serialized schedule replacement flags are invalid")

    digest = hashlib.sha256()
    digest.update(("{\"seed\":" + str(seed) + ",\"steps\":[").encode("ascii"))
    for index in range(steps):
        rows = tuple(int(value) for value in batch_rows[index].tolist())
        if len(set(rows)) != len(rows):
            raise FixedControlCallerError("serialized schedule contains duplicate rows within a batch")
        step = ScheduleStep(
            step=index,
            batch_global_rows=rows,
            draw_record_slots=tuple(int(value) for value in record_slots[index].tolist()),
            draw_position_slots=tuple(int(value) for value in positions[index].tolist()),
            used_replacement=bool(int(replacement[index].item())),
        )
        if index:
            digest.update(b",")
        digest.update(_schedule_step_bytes(step))
    digest.update(b"]}")
    semantic_sha256 = digest.hexdigest()
    metadata_semantic = str(metadata.get("schedule_semantic_sha256", ""))
    if semantic_sha256 != metadata_semantic:
        raise FixedControlCallerError("serialized schedule semantic digest differs")
    if expected_valid_mask_semantic_sha256 is not None and str(metadata.get("valid_mask_semantic_sha256", "")) != expected_valid_mask_semantic_sha256:
        raise FixedControlCallerError("serialized schedule valid-mask binding differs")
    return SerializedSchedule(
        path=schedule_path,
        file_sha256=actual_file_sha256,
        bank=bank,
        seed=seed,
        steps=steps,
        record_batch_size=record_batch_size,
        position_budget=position_budget,
        sequence_tokens=sequence_tokens,
        semantic_sha256=semantic_sha256,
        valid_mask_semantic_sha256=str(metadata.get("valid_mask_semantic_sha256", "")),
        batch_record_indices=batch_rows,
        draw_record_slots=record_slots,
        draw_position_slots=positions,
        used_replacement=replacement,
    )

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



CheckpointCallback = Callable[[Mapping[str, Any], nn.Module, ReadoutHook], Mapping[str, Any] | None]


def compose_checkpoint_callbacks(*callbacks: CheckpointCallback | None) -> CheckpointCallback:
    """Compose serializers/diagnostics without changing runner selection.

    ``run_training`` evaluates and freezes the validation selection metric before
    invoking its checkpoint callback.  This helper only merges disjoint state
    bindings, and rejects any callback that tries to inject a selection metric.
    """

    active = tuple(callback for callback in callbacks if callback is not None)
    if not active:
        raise FixedControlCallerError("at least one checkpoint callback is required")

    def callback(
        point: Mapping[str, Any],
        decoder: nn.Module,
        hook: ReadoutHook,
    ) -> Mapping[str, Any]:
        merged: dict[str, Any] = {}
        for child in active:
            value = child(point, decoder, hook)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise FixedControlCallerError("checkpoint callback must return a mapping")
            if "selection_metric" in value or DOMAIN_BALANCED_SELECTION_METRIC in value:
                raise FixedControlCallerError("diagnostic callback cannot rewrite selection")
            overlap = set(merged).intersection(value)
            if overlap:
                raise FixedControlCallerError(
                    f"checkpoint callback bindings overlap: {sorted(overlap)!r}"
                )
            merged.update(dict(value))
        return merged

    return callback


def make_fitting_metric_callback(
    *,
    source: BatchSource,
    embedding: torch.Tensor,
    fixed_rows: Sequence[int],
    full_bank_rows: Sequence[int],
    final_step: int,
    expected_sequence_tokens: int,
    expected_hidden_size: int,
    record_batch_size: int,
    position_budget: int,
    activation_dtype: torch.dtype | None = None,
    compute_base_logits: bool = True,
    fixed_row_count: int = 64,
) -> CheckpointCallback:
    """Create a read-only fitting diagnostic callback.

    The callback evaluates the immutable ``fixed_rows`` at every checkpoint
    callback (the caller supplies the frozen 64-row artifact) and evaluates
    ``full_bank_rows`` only at steps zero and ``final_step``.  It records the
    exact denominator, batch/chunk counts, full-vocabulary row count, and
    elapsed scope time.  Its result is nested under ``fitting_diagnostics`` by
    :func:`compose_checkpoint_callbacks`; no selection metric is returned or
    altered.
    """

    fixed = tuple(int(row) for row in fixed_rows)
    full = tuple(int(row) for row in full_bank_rows)
    if len(fixed) != int(fixed_row_count):
        raise FixedControlCallerError(
            f"frozen fitting diagnostic must contain exactly {fixed_row_count} rows"
        )
    if not full:
        raise FixedControlCallerError("full-bank diagnostic rows cannot be empty")
    if any(row < 0 for row in fixed + full) or len(set(fixed)) != len(fixed) or len(set(full)) != len(full):
        raise FixedControlCallerError("fitting diagnostic rows must be unique and nonnegative")
    if not set(fixed).issubset(set(full)):
        raise FixedControlCallerError("frozen diagnostic rows must be drawn from the full-bank rows")
    if isinstance(final_step, bool) or not isinstance(final_step, int) or final_step <= 0:
        raise FixedControlCallerError("fitting diagnostic final step is invalid")
    if record_batch_size <= 0 or position_budget <= 0:
        raise FixedControlCallerError("fitting diagnostic geometry is invalid")
    if len(fixed) % record_batch_size or len(full) % record_batch_size:
        raise FixedControlCallerError("fitting diagnostic rows must be batch aligned")

    def batches(rows: Sequence[int]) -> Iterator[BatchLike]:
        for start in range(0, len(rows), record_batch_size):
            chunk = tuple(rows[start : start + record_batch_size])
            if len(chunk) != record_batch_size:
                raise FixedControlCallerError("fitting diagnostic batch is incomplete")
            yield source.batch_for_global_rows(chunk)

    def evaluate_scope(
        scope: str,
        rows: Sequence[int],
        decoder: nn.Module,
        hook: FixedPublicReadoutHook,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        metrics = evaluate_batches(
            decoder,
            hook,
            batches(rows),
            embedding=embedding,
            expected_sequence_tokens=expected_sequence_tokens,
            expected_hidden_size=expected_hidden_size,
            expected_batch_records=record_batch_size,
            position_budget=position_budget,
            activation_dtype=activation_dtype,
            compute_base_logits=compute_base_logits,
        )
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise FixedControlCallerError("fitting diagnostic elapsed time is invalid")
        return {
            "scope": scope,
            "row_count": len(rows),
            "batch_count": int(metrics.get("batch_count", 0)),
            "position_chunk_count": int(metrics.get("position_chunk_count", 0)),
            "full_vocab_logits_rows": int(metrics.get("full_vocab_logits_rows", metrics["token_rows"])),
            "metrics": dict(metrics),
            "elapsed_seconds": float(elapsed),
            "timing_boundary": "includes random-access batch load, full-vocabulary projection, synchronization, and metric reduction",
        }

    def callback(
        point: Mapping[str, Any],
        decoder: nn.Module,
        hook: ReadoutHook,
    ) -> Mapping[str, Any]:
        try:
            step = int(point["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FixedControlCallerError("fitting diagnostic checkpoint lacks step") from exc
        scopes = [evaluate_scope("frozen64", fixed, decoder, hook)]
        if step in (0, final_step):
            scopes.append(evaluate_scope("full_bank", full, decoder, hook))
        return {
            "fitting_diagnostics": {
                "schema": "token-reconstruction.trr-p09-fitting-diagnostics.v1",
                "step": step,
                "selection_metric_untouched": True,
                "scopes": scopes,
            }
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
    diagnostic_runs: list[dict[str, Any]] = []
    for point in curve:
        if not isinstance(point, Mapping):
            raise FixedControlCallerError("learning curve point is not a mapping")
        metrics = point.get("validation")
        if isinstance(metrics, Mapping):
            rows = metrics.get("token_rows")
            if isinstance(rows, int) and not isinstance(rows, bool):
                validation_rows += rows
        binding = point.get("state_binding")
        diagnostics = binding.get("fitting_diagnostics") if isinstance(binding, Mapping) else None
        if diagnostics is None:
            continue
        if not isinstance(diagnostics, Mapping) or diagnostics.get("selection_metric_untouched") is not True:
            raise FixedControlCallerError("fitting diagnostics are not selection-isolated")
        scopes = diagnostics.get("scopes")
        if not isinstance(scopes, Sequence):
            raise FixedControlCallerError("fitting diagnostic scopes are missing")
        for scope in scopes:
            if not isinstance(scope, Mapping):
                raise FixedControlCallerError("fitting diagnostic scope is not a mapping")
            elapsed = scope.get("elapsed_seconds")
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
                raise FixedControlCallerError("fitting diagnostic elapsed time is invalid")
            row_count = scope.get("row_count")
            batch_count = scope.get("batch_count")
            chunk_count = scope.get("position_chunk_count")
            logits_rows = scope.get("full_vocab_logits_rows")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (row_count, batch_count, chunk_count, logits_rows)):
                raise FixedControlCallerError("fitting diagnostic denominator/cost fields are invalid")
            diagnostic_runs.append({
                "step": int(diagnostics.get("step", point.get("step", -1))),
                "scope": str(scope.get("scope", "")),
                "row_count": int(row_count),
                "batch_count": int(batch_count),
                "position_chunk_count": int(chunk_count),
                "full_vocab_logits_rows": int(logits_rows),
                "elapsed_seconds": float(elapsed),
            })
    diagnostic_cost = {
        "scope_run_count": len(diagnostic_runs),
        "total_elapsed_seconds": sum(item["elapsed_seconds"] for item in diagnostic_runs),
        "full_bank_steps": [item["step"] for item in diagnostic_runs if item["scope"] == "full_bank"],
        "runs": diagnostic_runs,
        "selection_metric_untouched": True,
    }
    return {
        "scope": "fixed_control_runner_process",
        "updates": steps,
        "position_draws_per_update": draws_per_step,
        "total_position_draws": total_draws,
        "validation_checkpoint_count": len(curve),
        "validation_token_rows_across_checkpoints": validation_rows,
        "fitting_diagnostics": diagnostic_cost,
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
    "compose_checkpoint_callbacks",
    "make_fitting_metric_callback",
    "inherited_schedule_steps",
    "join_public_validation_labels",
    "make_domain_validation_callback",
    "make_fixed_checkpoint_callback",
    "materialize_schedule_plan",
    "SerializedSchedule",
    "load_serialized_schedule",
    "signed_p09_checkpoint_grid",
    "write_fixed_control_receipt",
]
