"""Deterministic public-data training machinery for TRR-P04.

This module keeps the S/H/D training schedule and all public evidence explicit.
It does not load evaluator observations, target-update weights, or private
truth. Candidate identities are generated once from a frozen public affine
resource and are consumed only by the H/D training objectives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn

from .p04_objectives import (
    DEFAULT_HARD_MARGIN,
    DEFAULT_HARD_WEIGHT,
    DEFAULT_RANK_WEIGHT,
    DEFAULT_STUDENT_TEMPERATURE,
    derive_rank_scale,
    student_objective,
)
from .p04_student import (
    ALL_METHODS,
    METHOD_AFFINE,
    METHOD_D,
    METHOD_H,
    METHOD_S,
    P04StudentError,
    StudentArchitectureConfig,
    build_student,
    initialize_student,
    normalize_public_embeddings,
    save_student_state,
    state_sha256,
    validate_embedding_table,
)


TASK_ID = "TRR-P04"
TRAINING_SCHEMA = "token-reconstruction.trr-p04-training.v1"
POOL_SCHEMA = "token-reconstruction.trr-p04-public-pool.v1"
SCHEDULE_SCHEMA = "token-reconstruction.trr-p04-position-schedule.v1"
CANDIDATE_SCHEMA = "token-reconstruction.trr-p04-candidate-identities.v1"
CANDIDATE_PREPARATION_SCHEMA = "token-reconstruction.trr-p04-candidate-preparation.v1"
TEACHER_EVIDENCE_SCHEMA = "token-reconstruction.trr-p04-teacher-evidence.v1"
BOS_TOKEN_ID = 128000
DEFAULT_STEPS = 3000
DEFAULT_RECORD_BATCH_SIZE = 8
DEFAULT_POSITION_BUDGET = 512
DEFAULT_REPLAY_FRACTION = 0.75
DEFAULT_VALIDATION_EVERY = 100
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_PROJECTION_CHUNK = 512
DEFAULT_CANDIDATE_K = 32
DEFAULT_MAX_HOST_RSS_GIB = 16.0


class P04TrainingError(RuntimeError):
    """Raised when a public P04 training contract cannot be established."""


def _regular_file(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P04TrainingError(f"{label} must be a regular file: {path}")
    return path


def file_sha256(path: Path) -> str:
    path = _regular_file(path, label="hashed asset")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_load(path: Path, *, label: str) -> Any:
    path = _regular_file(path, label=label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P04TrainingError(f"cannot parse {label}: {path}") from exc


def _metadata(path: Path) -> dict[str, str]:
    try:
        with safe_open(str(_regular_file(path, label="safetensors asset")), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception as exc:
        raise P04TrainingError(f"cannot inspect safetensors metadata: {path}") from exc


def _load_component(path: Path, key: str, *, label: str) -> torch.Tensor:
    path = _regular_file(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in set(handle.keys()):
                raise P04TrainingError(f"{label} is missing tensor {key!r}")
            return handle.get_tensor(key).contiguous()
    except P04TrainingError:
        raise
    except Exception as exc:
        raise P04TrainingError(f"cannot load {label}: {path}") from exc


def _load_any_component(path: Path, keys: Sequence[str], *, label: str) -> tuple[torch.Tensor, str]:
    path = _regular_file(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in keys:
                if key in available:
                    return handle.get_tensor(key).contiguous(), key
    except Exception as exc:
        raise P04TrainingError(f"cannot load {label}: {path}") from exc
    raise P04TrainingError(f"{label} contains none of the expected tensors {tuple(keys)}")


def _records_payload(path: Path, *, label: str) -> list[dict[str, Any]]:
    payload = _json_load(path, label=label)
    records = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list) or not records:
        raise P04TrainingError(f"{label} must be a non-empty record list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(records):
        if not isinstance(value, Mapping) or not isinstance(value.get("record_id"), str):
            raise P04TrainingError(f"{label} record {index} has no string record_id")
        record = dict(value)
        record_id = str(record["record_id"])
        if not record_id or record_id in seen:
            raise P04TrainingError(f"{label} record IDs are empty or duplicated")
        seen.add(record_id)
        result.append(record)
    return result


def _record_style(record: Mapping[str, Any]) -> str:
    for key in ("style", "group", "source", "dataset", "domain"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return "public"


def _record_length(record: Mapping[str, Any], *, positions: int) -> int | None:
    for key in ("full_token_count", "sequence_length"):
        value = record.get(key)
        if value is not None:
            return int(value)
    value = record.get("post_bos_token_count")
    if value is not None:
        return int(value) + 1
    return None


def _validate_mask(value: torch.Tensor, *, rows: int, positions: int, label: str) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (rows, positions):
        raise P04TrainingError(f"{label} must have shape [{rows},{positions}]")
    if value.dtype not in (torch.bool, torch.uint8):
        raise P04TrainingError(f"{label} must be boolean")
    result = value.to(dtype=torch.bool).contiguous()
    if not result[:, 0].all().item() or not result[:, 1:].any(dim=1).all().item():
        raise P04TrainingError(f"{label} must include BOS and one post-BOS position")
    if not torch.equal(result, result.cumprod(dim=1).to(torch.bool)):
        raise P04TrainingError(f"{label} must be right-padded")
    return result


def _derive_mask(records: Sequence[Mapping[str, Any]], *, rows: int, positions: int) -> torch.Tensor:
    if len(records) != rows:
        raise P04TrainingError("record metadata does not match observation rows")
    mask = torch.ones((rows, positions), dtype=torch.bool)
    for row, record in enumerate(records):
        count = _record_length(record, positions=positions)
        if count is None:
            continue
        if count < 2 or count > positions:
            raise P04TrainingError(f"record {record['record_id']} has invalid sequence length {count}")
        mask[row, count:] = False
    return mask


def _validate_observations(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 1 or value.shape[2] <= 0:
        raise P04TrainingError(f"{label} must be [records,time>1,hidden]")
    if not value.dtype.is_floating_point or not torch.isfinite(value).all().item():
        raise P04TrainingError(f"{label} must be finite floating point")
    return value.contiguous()


def _validate_labels(value: torch.Tensor, *, rows: int, positions: int, mask: torch.Tensor, vocab_size: int, label: str, bos_token_id: int = BOS_TOKEN_ID) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (rows, positions):
        raise P04TrainingError(f"{label} must have shape [{rows},{positions}]")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise P04TrainingError(f"{label} must be integer token IDs")
    result = value.to(dtype=torch.long).contiguous()
    if result[:, 0].ne(int(bos_token_id)).any().item():
        raise P04TrainingError(f"{label} rows must begin with BOS token {bos_token_id}")
    scored_mask = mask.clone()
    scored_mask[:, 0] = False
    scored = result[scored_mask]
    if scored.lt(0).any().item() or scored.ge(vocab_size).any().item():
        raise P04TrainingError(f"{label} has out-of-range active IDs")
    return result


@dataclass(frozen=True)
class PublicPool:
    """One public pool with activation, label, mask, and identity metadata."""

    observations: torch.Tensor
    labels: torch.Tensor
    valid_mask: torch.Tensor
    record_ids: tuple[str, ...]
    styles: tuple[str, ...]
    source_path: str
    source_sha256: str
    records_path: str
    records_sha256: str

    @property
    def rows(self) -> int:
        return int(self.observations.shape[0])

    @property
    def positions(self) -> int:
        return int(self.observations.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.observations.shape[2])

    @property
    def post_bos_positions(self) -> int:
        return int(self.valid_mask[:, 1:].sum().item())


def load_public_pool(
    observation_path: Path,
    records_path: Path,
    *,
    truth_path: Path | None = None,
    mask_path: Path | None = None,
    embedding_vocab_size: int | None = None,
    bos_token_id: int = BOS_TOKEN_ID,
) -> PublicPool:
    """Load a public training pool from separate or combined safetensors.

    Combined PR7 artifacts use ``activations``, ``token_ids``, and
    ``attention_mask``. Separate files use the same key names. Truth is read
    only after activation geometry and record identities have passed checks.
    """

    records = _records_payload(records_path, label="public record manifest")
    observations, observation_key = _load_any_component(
        observation_path, ("activations", "observations"), label="public observations"
    )
    observations = _validate_observations(observations, label="public observations")
    rows, positions, hidden = map(int, observations.shape)
    if rows != len(records):
        raise P04TrainingError("public record manifest does not match observation rows")
    if mask_path is None:
        try:
            mask = _load_component(observation_path, "attention_mask", label="public attention mask")
        except P04TrainingError:
            mask = _derive_mask(records, rows=rows, positions=positions)
    else:
        mask = _load_component(mask_path, "valid_mask", label="public valid mask")
    mask = _validate_mask(mask, rows=rows, positions=positions, label="public valid mask")
    if truth_path is None:
        labels = _load_component(observation_path, "token_ids", label="public labels")
    else:
        labels = _load_component(truth_path, "token_ids", label="public labels")
    if embedding_vocab_size is None:
        active_max = int(labels[mask].max().item())
        embedding_vocab_size = max(active_max + 1, 1)
    labels = _validate_labels(
        labels,
        rows=rows,
        positions=positions,
        mask=mask,
        vocab_size=int(embedding_vocab_size),
        label="public labels",
        bos_token_id=bos_token_id,
    )
    ids = tuple(str(record["record_id"]) for record in records)
    styles = tuple(_record_style(record) for record in records)
    return PublicPool(
        observations=observations,
        labels=labels,
        valid_mask=mask,
        record_ids=ids,
        styles=styles,
        source_path=str(Path(observation_path).expanduser().resolve()),
        source_sha256=file_sha256(observation_path),
        records_path=str(Path(records_path).expanduser().resolve()),
        records_sha256=file_sha256(records_path),
    )


def combine_public_pools(replay: PublicPool, correction: PublicPool) -> PublicPool:
    """Concatenate equal-geometry public pools with a collision check."""

    if (replay.positions, replay.hidden_size) != (correction.positions, correction.hidden_size):
        raise P04TrainingError("replay and correction geometries differ; pad in setup first")
    overlap = set(replay.record_ids).intersection(correction.record_ids)
    if overlap:
        raise P04TrainingError(f"replay/correction record overlap: {sorted(overlap)[:3]}")
    return PublicPool(
        observations=torch.cat((replay.observations, correction.observations), dim=0),
        labels=torch.cat((replay.labels, correction.labels), dim=0),
        valid_mask=torch.cat((replay.valid_mask, correction.valid_mask), dim=0),
        record_ids=replay.record_ids + correction.record_ids,
        styles=replay.styles + correction.styles,
        source_path=f"{replay.source_path}|{correction.source_path}",
        source_sha256=canonical_hash([replay.source_sha256, correction.source_sha256]),
        records_path=f"{replay.records_path}|{correction.records_path}",
        records_sha256=canonical_hash([replay.records_sha256, correction.records_sha256]),
    )


def load_embedding_table(path: Path, *, hidden_size: int, vocab_size: int) -> torch.Tensor:
    table, key = _load_any_component(path, ("embeddings", "embedding_table"), label="public normalized embeddings")
    if key != "embeddings":
        table = table
    try:
        validate_embedding_table(table, hidden_size=hidden_size, vocab_size=vocab_size, require_unit_norm=True)
    except P04StudentError as exc:
        raise P04TrainingError("public embedding table failed normalized geometry validation") from exc
    return table.float().contiguous()


def _allocate_positions(
    valid_count: int,
    *,
    budget: int,
    required: Sequence[int] = (),
) -> list[int]:
    """Select deterministic active post-BOS columns, preserving required rows."""

    available = list(range(1, valid_count))
    required_set = sorted({int(value) for value in required})
    if any(value not in available for value in required_set):
        raise P04TrainingError("teacher evidence position is outside its public record")
    if len(required_set) > budget:
        raise P04TrainingError("teacher evidence exceeds per-record position budget")
    if len(available) <= budget:
        return available
    chosen = set(required_set)
    remaining = [value for value in available if value not in chosen]
    need = budget - len(chosen)
    if need:
        # Even spacing gives every length region a fixed opportunity while
        # required teacher rows are always retained.
        indices = torch.linspace(0, len(remaining) - 1, steps=need).round().to(torch.long).tolist()
        for index in indices:
            chosen.add(remaining[int(index)])
    return sorted(chosen)


@dataclass(frozen=True)
class PositionSchedule:
    """One immutable record/position schedule shared by every arm."""

    record_indices: torch.Tensor  # [steps,batch]
    selected_mask: torch.Tensor  # [steps,batch,time]
    seed: int
    record_batch_size: int
    position_budget: int
    replay_fraction: float
    schedule_sha256: str

    @property
    def steps(self) -> int:
        return int(self.record_indices.shape[0])


def make_position_schedule(
    pool: PublicPool,
    *,
    replay_records: int,
    steps: int = DEFAULT_STEPS,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    position_budget: int = DEFAULT_POSITION_BUDGET,
    replay_fraction: float = DEFAULT_REPLAY_FRACTION,
    seed: int,
    required_positions: Mapping[str, Sequence[int]] | None = None,
) -> PositionSchedule:
    """Create a deterministic 75/25 record schedule and masked positions."""

    if replay_records <= 0 or replay_records >= pool.rows:
        raise P04TrainingError("replay/correction pool split is empty")
    if steps <= 0 or record_batch_size <= 0 or position_budget <= 0:
        raise P04TrainingError("schedule dimensions must be positive")
    if not (0.0 < replay_fraction < 1.0):
        raise P04TrainingError("replay fraction must be strictly between zero and one")
    replay_count = round(record_batch_size * replay_fraction)
    correction_count = record_batch_size - replay_count
    if replay_count <= 0 or correction_count <= 0:
        raise P04TrainingError("record batch must contain both replay and correction rows")
    required_positions = required_positions or {}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    replay_perm = torch.randperm(replay_records, generator=generator).tolist()
    correction_perm = torch.randperm(pool.rows - replay_records, generator=generator).tolist()
    replay_cursor = 0
    correction_cursor = 0
    batches: list[list[int]] = []
    masks: list[torch.Tensor] = []
    for _step in range(steps):
        chosen: list[int] = []
        for count, perm, cursor, offset in (
            (replay_count, replay_perm, replay_cursor, 0),
            (correction_count, correction_perm, correction_cursor, replay_records),
        ):
            for _ in range(count):
                if cursor >= len(perm):
                    reshuffle = torch.randperm(len(perm), generator=generator).tolist()
                    perm[:] = reshuffle
                    cursor = 0
                chosen.append(offset + int(perm[cursor]))
                cursor += 1
            if offset == 0:
                replay_cursor = cursor
            else:
                correction_cursor = cursor
        # A seed determines row order; sort is deliberately not applied.
        batches.append(chosen)
        selected = torch.zeros((record_batch_size, pool.positions), dtype=torch.bool)
        for local, global_index in enumerate(chosen):
            record_id = pool.record_ids[global_index]
            valid_count = int(pool.valid_mask[global_index].sum().item())
            positions = _allocate_positions(
                valid_count,
                budget=max(1, position_budget // record_batch_size),
                required=required_positions.get(record_id, ()),
            )
            selected[local, positions] = True
        if int(selected.sum().item()) > position_budget:
            raise P04TrainingError("schedule exceeded selected position budget")
        if (selected & ~pool.valid_mask[torch.tensor(chosen)]).any().item() or selected[:, 0].any().item():
            raise P04TrainingError("schedule selected invalid/BOS position")
        masks.append(selected)
    record_indices = torch.tensor(batches, dtype=torch.long)
    selected_mask = torch.stack(masks).contiguous()
    digest = canonical_hash({"record_indices": tensor_sha256(record_indices), "selected_mask": tensor_sha256(selected_mask)})
    return PositionSchedule(
        record_indices=record_indices,
        selected_mask=selected_mask,
        seed=int(seed),
        record_batch_size=int(record_batch_size),
        position_budget=int(position_budget),
        replay_fraction=float(replay_fraction),
        schedule_sha256=digest,
    )


def save_schedule(path: Path, schedule: PositionSchedule, *, pool: PublicPool, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise P04TrainingError(f"schedule artifact is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "record_indices": schedule.record_indices,
            "selected_mask": schedule.selected_mask,
        },
        str(path),
        metadata={
            "schema": SCHEDULE_SCHEMA,
            "task_id": TASK_ID,
            "seed": str(schedule.seed),
            "record_batch_size": str(schedule.record_batch_size),
            "position_budget": str(schedule.position_budget),
            "replay_fraction": str(schedule.replay_fraction),
            "pool_record_order_sha256": canonical_hash(list(pool.record_ids)),
            "schedule_sha256": schedule.schedule_sha256,
            **{str(key): str(value) for key, value in (metadata or {}).items()},
        },
    )
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path), "schedule_sha256": schedule.schedule_sha256}


def _deterministic_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k with ascending token IDs for exact boundary ties."""

    if logits.ndim != 2 or k <= 0 or k > logits.shape[1]:
        raise P04TrainingError("top-k logits geometry is invalid")
    values, ids = torch.topk(logits, k=k, dim=1, largest=True, sorted=False)
    threshold = values.amin(dim=1)
    chosen_rows: list[torch.Tensor] = []
    for row in range(int(logits.shape[0])):
        row_logits = logits[row]
        row_ids = ids[row]
        row_threshold = threshold[row]
        greater = torch.nonzero(row_logits > row_threshold, as_tuple=False).reshape(-1)
        equal = torch.nonzero(row_logits == row_threshold, as_tuple=False).reshape(-1)
        need = k - int(greater.numel())
        if need < 0:
            raise P04TrainingError("top-k threshold count is inconsistent")
        if need:
            equal = equal[:need]
        selected = torch.cat((greater, equal), dim=0)
        if selected.numel() != k:
            # No exact boundary tie: topk's selected IDs are already enough.
            selected = row_ids
        selected_scores = row_logits[selected]
        # Stable ID sort followed by stable descending score sort implements
        # descending scores and ascending IDs among equal values.
        id_order = selected.argsort(stable=True)
        selected = selected[id_order]
        score_order = row_logits[selected].argsort(descending=True, stable=True)
        chosen_rows.append(selected[score_order])
    return torch.stack(chosen_rows, dim=0).to(dtype=torch.int32)


def generate_candidate_ids(
    pool: PublicPool,
    embedding_table: torch.Tensor,
    *,
    affine_state: Mapping[str, torch.Tensor] | None,
    config: StudentArchitectureConfig,
    device: torch.device,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    projection_chunk: int = DEFAULT_PROJECTION_CHUNK,
) -> torch.Tensor:
    """Generate fixed public A1 candidate identities before fitting."""

    if candidate_k <= 0 or candidate_k > int(embedding_table.shape[0]):
        raise P04TrainingError("candidate budget is outside vocabulary")
    validate_embedding_table(
        embedding_table,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        require_unit_norm=True,
    )
    model = build_student(METHOD_AFFINE, config=config, affine_state=affine_state).to(device).eval()
    table = embedding_table.to(device=device, dtype=torch.float32)
    result = torch.empty((pool.rows, pool.positions, candidate_k), dtype=torch.int32)
    with torch.inference_mode():
        for start in range(0, pool.rows, record_batch_size):
            stop = min(start + record_batch_size, pool.rows)
            activation = pool.observations[start:stop].to(device=device, dtype=torch.float32)
            hidden = model.projected_hidden(activation).reshape(-1, config.hidden_size)
            for chunk_start in range(0, int(hidden.shape[0]), projection_chunk):
                chunk_stop = min(chunk_start + projection_chunk, int(hidden.shape[0]))
                logits = hidden[chunk_start:chunk_stop] @ table.transpose(0, 1)
                logits = logits * model.logit_scale.float().exp()
                result[start:stop].reshape(-1, candidate_k)[chunk_start:chunk_stop] = _deterministic_topk(logits, candidate_k).cpu()
    return result


def save_candidate_ids(path: Path, candidate_ids: torch.Tensor, *, pool: PublicPool, embedding_sha256: str, affine_provenance: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise P04TrainingError(f"candidate artifact is create-only: {path}")
    if candidate_ids.ndim != 3 or candidate_ids.shape[:2] != (pool.rows, pool.positions):
        raise P04TrainingError("candidate artifact geometry changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"candidate_ids": candidate_ids.to(dtype=torch.int32).contiguous()},
        str(path),
        metadata={
            "schema": CANDIDATE_SCHEMA,
            "task_id": TASK_ID,
            "candidate_k": str(candidate_ids.shape[2]),
            "pool_record_order_sha256": canonical_hash(list(pool.record_ids)),
            "pool_observation_sha256": pool.source_sha256,
            "embedding_sha256": embedding_sha256,
            "affine_provenance_json": json.dumps(dict(affine_provenance), sort_keys=True),
        },
    )
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path), "tensor_sha256": tensor_sha256(candidate_ids)}


@dataclass(frozen=True)
class TeacherEvidence:
    candidate_ids: torch.Tensor
    teacher_scores: torch.Tensor
    record_ids: tuple[str, ...]
    positions: tuple[int, ...]
    row_kind: tuple[str, ...]
    sigma_q: float
    tie_tolerance: float
    source_path: str
    source_sha256: str
    metadata: Mapping[str, Any]

    @property
    def rows(self) -> int:
        return int(self.candidate_ids.shape[0])


def load_teacher_evidence(path: Path, *, expected_candidate_k: int = DEFAULT_CANDIDATE_K) -> TeacherEvidence:
    path = _regular_file(path, label="teacher evidence")
    candidates = _load_component(path, "candidate_ids", label="teacher candidate IDs").to(dtype=torch.int64)
    scores = _load_component(path, "teacher_scores", label="teacher scores").float()
    if candidates.ndim != 2 or candidates.shape != scores.shape or candidates.shape[1] != expected_candidate_k:
        raise P04TrainingError("teacher evidence candidate/score geometry changed")
    if not torch.isfinite(scores).all().item():
        raise P04TrainingError("teacher evidence contains non-finite scores")
    meta = _metadata(path)
    rows_payload = meta.get("rows_json")
    if not rows_payload:
        raise P04TrainingError("teacher evidence metadata lacks row identities")
    try:
        row_values = json.loads(rows_payload)
    except json.JSONDecodeError as exc:
        raise P04TrainingError("teacher evidence rows_json is malformed") from exc
    if not isinstance(row_values, list) or len(row_values) != int(candidates.shape[0]):
        raise P04TrainingError("teacher evidence row identities do not match arrays")
    record_ids: list[str] = []
    positions: list[int] = []
    kinds: list[str] = []
    for row in row_values:
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
            raise P04TrainingError("teacher evidence row lacks record ID")
        record_ids.append(str(row["record_id"]))
        positions.append(int(row["position"]))
        kinds.append(str(row.get("kind", "qualified")))
    sigma_raw = meta.get("sigma_q")
    tie_raw = meta.get("tie_tolerance")
    if sigma_raw is None:
        scale = derive_rank_scale(candidates, scores)
        sigma_q = float(scale["sigma_q"])
        tie_tolerance = float(scale["tie_tolerance"])
    else:
        sigma_q = float(sigma_raw)
        tie_tolerance = float(tie_raw) if tie_raw is not None else max(1e-6, 0.01 * sigma_q)
    if not math.isfinite(sigma_q) or sigma_q <= 0 or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise P04TrainingError("teacher evidence score scale is invalid")
    return TeacherEvidence(
        candidate_ids=candidates.contiguous(),
        teacher_scores=scores.contiguous(),
        record_ids=tuple(record_ids),
        positions=tuple(positions),
        row_kind=tuple(kinds),
        sigma_q=sigma_q,
        tie_tolerance=tie_tolerance,
        source_path=str(path),
        source_sha256=file_sha256(path),
        metadata=meta,
    )


def _teacher_lookup(pool: PublicPool, evidence: TeacherEvidence) -> dict[tuple[int, int], int]:
    index = {record_id: row for row, record_id in enumerate(pool.record_ids)}
    lookup: dict[tuple[int, int], int] = {}
    for evidence_row, (record_id, position) in enumerate(zip(evidence.record_ids, evidence.positions)):
        if record_id not in index:
            raise P04TrainingError(f"teacher evidence record is outside training pools: {record_id}")
        key = (index[record_id], int(position))
        if key in lookup:
            raise P04TrainingError("teacher evidence row identities are duplicated")
        lookup[key] = evidence_row
    return lookup


def build_teacher_arrays(
    pool: PublicPool,
    candidate_ids: torch.Tensor,
    evidence: TeacherEvidence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Bind qualified teacher rows to the frozen full-pool candidate table."""

    if candidate_ids.shape[:2] != (pool.rows, pool.positions):
        raise P04TrainingError("candidate table does not match public pool")
    lookup = _teacher_lookup(pool, evidence)
    teacher_scores = torch.zeros((pool.rows, pool.positions, candidate_ids.shape[2]), dtype=torch.float32)
    teacher_mask = torch.zeros((pool.rows, pool.positions), dtype=torch.bool)
    for key, evidence_row in lookup.items():
        row, position = key
        if position <= 0 or position >= pool.positions or not pool.valid_mask[row, position].item():
            raise P04TrainingError("teacher evidence position is inactive or BOS")
        expected = candidate_ids[row, position].to(torch.int64)
        supplied = evidence.candidate_ids[evidence_row].to(torch.int64)
        if not torch.equal(expected, supplied):
            raise P04TrainingError(f"candidate identity mismatch at {pool.record_ids[row]}:{position}")
        teacher_scores[row, position] = evidence.teacher_scores[evidence_row]
        teacher_mask[row, position] = True
    required = {
        pool.record_ids[row]: [position for (row_index, position), _ in lookup.items() if row_index == row]
        for row in range(pool.rows)
        if any(row_index == row for row_index, _ in lookup)
    }
    return candidate_ids, teacher_scores, teacher_mask, {
        "evidence_rows": evidence.rows,
        "evidence_record_count": len(set(evidence.record_ids)),
        "required_positions": required,
        "sigma_q": evidence.sigma_q,
        "tie_tolerance": evidence.tie_tolerance,
        "source_sha256": evidence.source_sha256,
    }


def _runtime_setup(*, seed: int, threads: int = 4, interop_threads: int = 1) -> None:
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(int(interop_threads))
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _projection_logits(model: nn.Module, activation: torch.Tensor, table: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    hidden = model.projected_hidden(activation)
    if selected.ndim != 2 or selected.shape != hidden.shape[:2] or not selected.any().item():
        raise P04TrainingError("selected projection mask is empty or misaligned")
    if (selected[:, 0]).any().item():
        raise P04TrainingError("BOS cannot be selected for training/scoring")
    rows = hidden[selected]
    logits = rows @ table.transpose(0, 1)
    scale = model.logit_scale.float().exp()
    if not torch.isfinite(scale).item() or scale.item() <= 0:
        raise P04TrainingError("student logit scale is invalid")
    return logits * scale


def _batch_teacher_arrays(
    batch_indices: torch.Tensor,
    selected: torch.Tensor,
    candidate_ids: torch.Tensor,
    teacher_scores: torch.Tensor | None,
    teacher_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if candidate_ids is None:
        return None, None, None
    batch_candidates = candidate_ids.index_select(0, batch_indices)[selected]
    if teacher_scores is None or teacher_mask is None:
        return batch_candidates, None, None
    batch_scores = teacher_scores.index_select(0, batch_indices)[selected]
    batch_rank_mask = teacher_mask.index_select(0, batch_indices)[selected]
    return batch_candidates, batch_scores, batch_rank_mask


def _metrics_from_predictions(predictions: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, Any]:
    scored = valid_mask.clone().to(dtype=torch.bool)
    scored[:, 0] = False
    correct = predictions.eq(labels) & scored
    total = int(scored.sum().item())
    correct_total = int(correct.sum().item())
    record_total = scored.sum(dim=1)
    record_correct = correct.sum(dim=1)
    exact = record_correct.eq(record_total)
    return {
        "token_rows": total,
        "correct_tokens": correct_total,
        "token_accuracy": correct_total / total if total else 0.0,
        "exact_records": int(exact.sum().item()),
        "record_count": int(predictions.shape[0]),
        "record_accuracy": float(exact.float().mean().item()),
        "per_record_correct": record_correct.tolist(),
        "per_record_total": record_total.tolist(),
    }


def evaluate_public(
    model: nn.Module,
    pool: PublicPool,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    projection_chunk: int = DEFAULT_PROJECTION_CHUNK,
) -> dict[str, Any]:
    """Full-vocabulary public metric with no candidate/teacher inputs."""

    model.eval()
    table = embedding_table.to(device=device, dtype=torch.float32)
    all_predictions = torch.full((pool.rows, pool.positions), -1, dtype=torch.int64)
    tie_counts = torch.zeros((pool.rows, pool.positions), dtype=torch.int32)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, pool.rows, record_batch_size):
            stop = min(start + record_batch_size, pool.rows)
            activation = pool.observations[start:stop].to(device=device, dtype=torch.float32)
            valid = pool.valid_mask[start:stop].to(device=device, dtype=torch.bool)
            projected = model.projected_hidden(activation)
            active = valid.clone()
            active[:, 0] = False
            coordinates = active.nonzero(as_tuple=False)
            flat = projected[coordinates[:, 0], coordinates[:, 1]]
            for chunk_start in range(0, int(flat.shape[0]), projection_chunk):
                chunk_stop = min(chunk_start + projection_chunk, int(flat.shape[0]))
                logits = flat[chunk_start:chunk_stop] @ table.transpose(0, 1)
                logits = logits * model.logit_scale.float().exp()
                maxima = logits.amax(dim=-1, keepdim=True)
                ids = logits.argmax(dim=-1).to(dtype=torch.int64)
                ties = logits.eq(maxima).sum(dim=-1).to(dtype=torch.int32)
                coords = coordinates[chunk_start:chunk_stop].cpu()
                all_predictions[start + coords[:, 0], coords[:, 1]] = ids.cpu()
                tie_counts[start + coords[:, 0], coords[:, 1]] = ties.cpu()
    metrics = _metrics_from_predictions(all_predictions, pool.labels, pool.valid_mask)
    groups: dict[str, dict[str, int]] = {}
    scored_mask = pool.valid_mask.clone()
    scored_mask[:, 0] = False
    for index, style in enumerate(pool.styles):
        row = groups.setdefault(style, {"correct_tokens": 0, "token_rows": 0})
        row["correct_tokens"] += int((all_predictions[index].eq(pool.labels[index]) & scored_mask[index]).sum().item())
        row["token_rows"] += int(scored_mask[index].sum().item())
    group_acc = {
        key: value["correct_tokens"] / value["token_rows"]
        for key, value in groups.items()
        if value["token_rows"]
    }
    metrics.update({
        "style_token_accuracy": group_acc,
        "style_balanced_token_accuracy": sum(group_acc.values()) / len(group_acc) if group_acc else 0.0,
        "tie_count_total": int(tie_counts[scored_mask].sum().item()),
        "evaluation_seconds": time.perf_counter() - started,
        "predictions": all_predictions,
        "tie_counts": tie_counts,
    })
    return metrics


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = DEFAULT_STEPS
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE
    position_budget: int = DEFAULT_POSITION_BUDGET
    replay_fraction: float = DEFAULT_REPLAY_FRACTION
    validation_every: int = DEFAULT_VALIDATION_EVERY
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM
    projection_chunk: int = DEFAULT_PROJECTION_CHUNK
    hard_weight: float = DEFAULT_HARD_WEIGHT
    hard_margin: float = DEFAULT_HARD_MARGIN
    rank_weight: float = DEFAULT_RANK_WEIGHT
    student_temperature: float = DEFAULT_STUDENT_TEMPERATURE

    def validate(self) -> None:
        if self.steps <= 0 or self.record_batch_size <= 0 or self.position_budget <= 0 or self.validation_every <= 0:
            raise P04TrainingError("training integer configuration must be positive")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0 or self.projection_chunk <= 0:
            raise P04TrainingError("training optimization configuration is invalid")


def train_arm(
    method_id: str,
    *,
    pool: PublicPool,
    validation: PublicPool,
    embedding_table: torch.Tensor,
    affine_state: Mapping[str, torch.Tensor] | None,
    schedule: PositionSchedule,
    candidate_ids: torch.Tensor | None,
    teacher_scores: torch.Tensor | None,
    teacher_mask: torch.Tensor | None,
    sigma_q: float | None,
    tie_tolerance: float | None,
    seed: int,
    config: TrainingConfig,
    architecture: StudentArchitectureConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one arm with fixed schedule and write selected/final states."""

    if method_id not in ALL_METHODS:
        raise P04TrainingError(f"unknown P04 arm: {method_id}")
    config.validate()
    if schedule.steps != config.steps:
        raise P04TrainingError("schedule steps do not match training config")
    if method_id in (METHOD_H, METHOD_D) and candidate_ids is None:
        raise P04TrainingError(f"{method_id} requires frozen candidate IDs")
    if method_id == METHOD_D and (teacher_scores is None or teacher_mask is None or sigma_q is None):
        raise P04TrainingError("student_d requires qualified teacher arrays")
    _runtime_setup(seed=seed)
    model = initialize_student(method_id, seed=seed, config=architecture, affine_state=affine_state).to(device)
    model.train()
    table = embedding_table.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    checkpoints = set(range(0, config.steps + 1, config.validation_every))
    checkpoints.add(config.steps)
    curve: list[dict[str, Any]] = []
    best_step = 0
    best_state = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()
    initial_validation = evaluate_public(model, validation, embedding_table, device=device, record_batch_size=config.record_batch_size, projection_chunk=config.projection_chunk)
    best_metric = float(initial_validation["style_balanced_token_accuracy"])
    curve.append({"step": 0, "validation": {key: value for key, value in initial_validation.items() if key not in ("predictions", "tie_counts")}, "learning_rate": config.learning_rate})
    for step_index in range(config.steps):
        model.train()
        batch_indices = schedule.record_indices[step_index]
        selected = schedule.selected_mask[step_index]
        activation = pool.observations.index_select(0, batch_indices).to(device=device, dtype=torch.float32)
        labels = pool.labels.index_select(0, batch_indices).to(device=device, dtype=torch.long)
        active = pool.valid_mask.index_select(0, batch_indices)
        if (selected & ~active).any().item() or selected[:, 0].any().item() or not selected.any().item():
            raise P04TrainingError("schedule selected invalid or empty rows")
        selected_device = selected.to(device=device)
        logits = _projection_logits(model, activation, table, selected_device)
        target = labels[selected_device]
        batch_candidates = batch_scores = batch_rank_mask = None
        if candidate_ids is not None:
            batch_candidates, batch_scores, batch_rank_mask = _batch_teacher_arrays(
                batch_indices,
                selected,
                candidate_ids,
                teacher_scores,
                teacher_mask,
            )
            batch_candidates = batch_candidates.to(device=device, dtype=torch.int64)
            if batch_scores is not None:
                batch_scores = batch_scores.to(device=device, dtype=torch.float32)
            if batch_rank_mask is not None:
                batch_rank_mask = batch_rank_mask.to(device=device, dtype=torch.bool)
        result = student_objective(
            logits,
            target,
            method_id=method_id,
            candidate_ids=batch_candidates,
            teacher_scores=batch_scores,
            rank_mask=batch_rank_mask,
            hard_weight=config.hard_weight,
            hard_margin=config.hard_margin,
            rank_weight=config.rank_weight,
            sigma_q=sigma_q,
            tie_tolerance=tie_tolerance,
            student_temperature=config.student_temperature,
        )
        if not torch.isfinite(result.total).item():
            raise P04TrainingError("P04 training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        parameters = list(model.parameters())
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all().item() for parameter in parameters):
            raise P04TrainingError("P04 gradient is non-finite")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
            raise P04TrainingError("P04 parameter became non-finite")
        if (step_index + 1) in checkpoints:
            validation_metrics = evaluate_public(model, validation, embedding_table, device=device, record_batch_size=config.record_batch_size, projection_chunk=config.projection_chunk)
            curve.append({
                "step": step_index + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": {**result.scalar_dict(), "gradient_norm": float(gradient_norm.detach().cpu().item()), "selected_rows": int(target.numel())},
                "validation": {key: value for key, value in validation_metrics.items() if key not in ("predictions", "tie_counts")},
            })
            metric = float(validation_metrics["style_balanced_token_accuracy"])
            if step_index + 1 > 0 and metric > best_metric:
                best_metric = metric
                best_step = step_index + 1
                best_state = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    final_metrics = evaluate_public(model, pool, embedding_table, device=device, record_batch_size=config.record_batch_size, projection_chunk=config.projection_chunk)
    final_state = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    final_state_path = output_dir / "final.safetensors"
    selected_state_path = output_dir / "selected.safetensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state, strict=True)
    selected_receipt = save_student_state(
        model,
        selected_state_path,
        method_id=method_id,
        seed=seed,
        config=architecture,
        metadata={
            "training_schema": TRAINING_SCHEMA,
            "selected_step": best_step,
            "schedule_sha256": schedule.schedule_sha256,
            "teacher_source": "qualified_public_only" if method_id == METHOD_D else "none",
        },
    )
    model.load_state_dict(final_state, strict=True)
    # ``final`` is deliberately serialized separately from selected state so a
    # post-fit diagnostic cannot overwrite the public-validation choice.
    final_receipt = save_student_state(
        model,
        final_state_path,
        method_id=method_id,
        seed=seed,
        config=architecture,
        metadata={"training_schema": TRAINING_SCHEMA, "selected_step": "final", "schedule_sha256": schedule.schedule_sha256},
    )
    curve_path = output_dir / "learning_curve.json"
    curve_path.write_text(json.dumps(curve, indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else value) + "\n", encoding="utf-8")
    return {
        "method_id": method_id,
        "seed": seed,
        "selected_step": best_step,
        "selected_validation_style_balanced_accuracy": best_metric,
        "selected_state": selected_receipt,
        "final_state": final_receipt,
        "learning_curve_path": str(curve_path.resolve()),
        "learning_curve_sha256": file_sha256(curve_path),
        "final_fit_metrics": {key: value for key, value in final_metrics.items() if key not in ("predictions", "tie_counts")},
        "wall_seconds": time.perf_counter() - started,
        "host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1024 if platform.system() != "Darwin" else 1),
    }


__all__ = [
    "ALL_METHODS",
    "BOS_TOKEN_ID",
    "CANDIDATE_PREPARATION_SCHEMA",
    "CANDIDATE_SCHEMA",
    "DEFAULT_CANDIDATE_K",
    "METHOD_AFFINE",
    "METHOD_D",
    "METHOD_H",
    "METHOD_S",
    "POOL_SCHEMA",
    "P04TrainingError",
    "PositionSchedule",
    "PublicPool",
    "SCHEDULE_SCHEMA",
    "TEACHER_EVIDENCE_SCHEMA",
    "TeacherEvidence",
    "TRAINING_SCHEMA",
    "TrainingConfig",
    "build_teacher_arrays",
    "canonical_hash",
    "combine_public_pools",
    "evaluate_public",
    "file_sha256",
    "generate_candidate_ids",
    "load_embedding_table",
    "load_public_pool",
    "load_teacher_evidence",
    "make_position_schedule",
    "save_candidate_ids",
    "save_schedule",
    "tensor_sha256",
    "train_arm",
]
