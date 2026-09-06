"""Joint TRR-0005 affine and activation-context decoder.

This module contains the deliberately small comparison registered for
TRR-0005.  Each arm is trained from the same identity affine initialisation
and receives the same current-position cross-entropy draws:

``joint_full_affine``
    Trainable ``W``, ``b`` and log scale only.
``affine_causal_h_attention128``
    The same trainable affine base plus a zero-output, one-head causal
    attention path over ``H_0, ..., H_i``.
``affine_trained_diagonal_attention128``
    The same attention path with a strict diagonal attention mask.  It sees
    ``H_i`` but no earlier activation.  Query/key gradients are consequently
    zero in this control once each valid query has exactly one allowed key;
    this is recorded by the runner and is not treated as equal effective
    capacity.

No arm consumes source tokens, guessed prefixes, future observations, a
public-prefix call, candidate simulations, or A2.  The implementation is
kept separate from the frozen-base TRR-0004 extension so that the joint
training contract cannot accidentally inherit a frozen affine base.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn


TASK_ID = "TRR-0005"
SCHEMA = "token-reconstruction.trr0005-joint-decoder.v1"
DATA_SCHEMA = "token-reconstruction.trr0005-public-fit-data.v1"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256
DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_SEQUENCE_LENGTH = 192
DEFAULT_CONTEXT_WIDTH = 128
DEFAULT_RECORD_BATCH_SIZE = 8
DEFAULT_POSITION_BUDGET = 512
DEFAULT_STEPS = 3000
DEFAULT_VALIDATION_EVERY = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_SEED = 4005

AFFINE_METHOD = "joint_full_affine"
CAUSAL_ATTENTION_METHOD = "affine_causal_h_attention128"
DIAGONAL_ATTENTION_METHOD = "affine_trained_diagonal_attention128"
METHODS = (AFFINE_METHOD, CAUSAL_ATTENTION_METHOD, DIAGONAL_ATTENTION_METHOD)
ATTENTION_SCORE_MODE_DOT_PRODUCT = "dot_product"
ATTENTION_SCORE_MODE_COSINE_SCALE4 = "cosine_scale4"
ATTENTION_SCORE_MODES = (
    ATTENTION_SCORE_MODE_DOT_PRODUCT,
    ATTENTION_SCORE_MODE_COSINE_SCALE4,
)
COSINE_ATTENTION_SCORE_SCALE = 4.0
AttentionMode = Literal["causal", "diagonal"]


class JointDecoderError(RuntimeError):
    """Raised when a TRR-0005 model or data contract is invalid."""


def file_sha256(path: Path) -> str:
    """Hash one regular file in bounded blocks."""

    if path.is_symlink() or not path.is_file():
        raise JointDecoderError(f"resource must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash shape, dtype and contiguous tensor bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"trr0005-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_method(method_id: str) -> str:
    if method_id not in METHODS:
        raise JointDecoderError(f"unknown TRR-0005 method: {method_id}")
    return method_id


def _deterministic_linear_init(linear: nn.Linear, generator: torch.Generator) -> None:
    """Initialise one attention projection from the shared seed stream.

    The three Q/K/V projections are initialised in a single deterministic
    stream seeded by 4005.  The zero output projection is set separately so
    every contextual arm starts exactly at its affine base.
    """

    fan_in, fan_out = linear.in_features, linear.out_features
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    with torch.no_grad():
        linear.weight.copy_(
            torch.rand(
                linear.weight.shape,
                dtype=linear.weight.dtype,
                generator=generator,
            )
            * (2.0 * bound)
            - bound
        )
        linear.bias.zero_()


class JointAffineAttentionDecoder(nn.Module):
    """One jointly trainable affine decoder, optionally with an H-only path."""

    def __init__(
        self,
        hidden_size: int,
        vocabulary_size: int,
        method_id: str,
        *,
        context_width: int = DEFAULT_CONTEXT_WIDTH,
        seed: int = DEFAULT_SEED,
        attention_score_mode: str = ATTENTION_SCORE_MODE_DOT_PRODUCT,
    ) -> None:
        super().__init__()
        _validate_method(method_id)
        if hidden_size <= 0 or vocabulary_size <= 0 or context_width <= 0:
            raise JointDecoderError("decoder geometry must be positive")
        self.hidden_size = int(hidden_size)
        self.vocabulary_size = int(vocabulary_size)
        self.context_width = int(context_width)
        self.method_id = method_id
        if method_id == AFFINE_METHOD:
            if attention_score_mode not in (
                ATTENTION_SCORE_MODE_DOT_PRODUCT,
                "none",
            ):
                raise JointDecoderError(
                    "the affine arm cannot use an attention score mode"
                )
            self.attention_score_mode = "none"
        else:
            if attention_score_mode not in ATTENTION_SCORE_MODES:
                raise JointDecoderError(
                    f"unknown attention score mode: {attention_score_mode}"
                )
            self.attention_score_mode = str(attention_score_mode)

        self.W = nn.Parameter(torch.eye(self.hidden_size, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

        if method_id == AFFINE_METHOD:
            self.query = None
            self.key = None
            self.value = None
            self.output = None
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            self.query = nn.Linear(self.hidden_size, self.context_width)
            self.key = nn.Linear(self.hidden_size, self.context_width)
            self.value = nn.Linear(self.hidden_size, self.context_width)
            self.output = nn.Linear(self.context_width, self.hidden_size)
            _deterministic_linear_init(self.query, generator)
            _deterministic_linear_init(self.key, generator)
            _deterministic_linear_init(self.value, generator)
            with torch.no_grad():
                self.output.weight.zero_()
                self.output.bias.zero_()

    @property
    def attention_mode(self) -> AttentionMode | None:
        if self.method_id == CAUSAL_ATTENTION_METHOD:
            return "causal"
        if self.method_id == DIAGONAL_ATTENTION_METHOD:
            return "diagonal"
        return None

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.s.float().exp()

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def _check_inputs(self, activation: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 3 or int(activation.shape[-1]) != self.hidden_size:
            raise JointDecoderError("activation must be [records, positions, hidden]")
        if not activation.dtype.is_floating_point:
            raise JointDecoderError("activation must be floating point")
        if not torch.isfinite(activation).all().item():
            raise JointDecoderError("activation contains non-finite values")
        if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(activation.shape[:2]):
            raise JointDecoderError("valid mask geometry does not match activations")
        if valid_mask.dtype not in (torch.bool, torch.uint8):
            raise JointDecoderError("valid mask must be boolean")
        mask = valid_mask.to(device=activation.device, dtype=torch.bool)
        if mask.shape[1] <= 1 or not mask[:, 0].all().item():
            raise JointDecoderError("every row must contain a valid BOS position")
        return mask

    def _added_path(self, activation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.attention_mode is None:
            return torch.zeros_like(activation.float())
        assert self.query is not None
        assert self.key is not None
        assert self.value is not None
        assert self.output is not None
        value = F.layer_norm(
            activation.float(),
            (self.hidden_size,),
            weight=None,
            bias=None,
            eps=1e-5,
        )
        query = self.query(value)
        key = self.key(value)
        projected_value = self.value(value)
        if self.attention_score_mode == ATTENTION_SCORE_MODE_COSINE_SCALE4:
            # The repair bounds score magnitude while retaining the same
            # trainable Q/K/V/output parameterization.  In the diagonal
            # control the allowed set still has one key, so its probability
            # remains exactly one and Q/K remain gradient-inactive.
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
            scores = (query @ key.transpose(-1, -2)) * COSINE_ATTENTION_SCORE_SCALE
        else:
            scores = query @ key.transpose(-1, -2) / math.sqrt(self.context_width)
        positions = torch.arange(int(activation.shape[1]), device=activation.device)
        if self.attention_mode == "causal":
            allowed = positions[None, :] <= positions[:, None]
        else:
            allowed = torch.eye(
                int(activation.shape[1]),
                dtype=torch.bool,
                device=activation.device,
            )
        allowed = allowed.unsqueeze(0) & mask[:, None, :]
        masked_scores = scores.masked_fill(~allowed, float("-inf"))
        valid_query = mask.unsqueeze(-1)
        has_key = allowed.any(dim=-1, keepdim=True)
        safe_scores = torch.where(has_key, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = torch.where(valid_query & has_key, weights, torch.zeros_like(weights))
        attended = weights @ projected_value
        output = self.output(attended)
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)

    def pre_normalized_hidden(
        self, activation: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        base = activation.float() @ self.W.float().T + self.b.float()
        combined = base + self._added_path(activation, mask)
        return torch.where(mask.unsqueeze(-1), combined, torch.zeros_like(combined))

    def projected_hidden(
        self, activation: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        projected = F.normalize(self.pre_normalized_hidden(activation, mask), dim=-1)
        return torch.where(mask.unsqueeze(-1), projected, torch.zeros_like(projected))

    def logits_from_rows(
        self,
        projected_hidden: torch.Tensor,
        record_slots: torch.Tensor,
        position_slots: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        if projected_hidden.ndim != 3 or int(projected_hidden.shape[-1]) != self.hidden_size:
            raise JointDecoderError("projected hidden geometry changed")
        if record_slots.ndim != 1 or position_slots.ndim != 1:
            raise JointDecoderError("draw indices must be vectors")
        if tuple(record_slots.shape) != tuple(position_slots.shape):
            raise JointDecoderError("draw index vectors differ")
        if not embedding_table.dtype.is_floating_point or embedding_table.ndim != 2:
            raise JointDecoderError("embedding table must be a floating matrix")
        if tuple(embedding_table.shape) != (self.vocabulary_size, self.hidden_size):
            raise JointDecoderError("embedding table geometry changed")
        rows = projected_hidden[record_slots.to(projected_hidden.device), position_slots.to(projected_hidden.device)]
        logits = rows.to(embedding_table.dtype) @ embedding_table.transpose(0, 1)
        result = logits.float() * self.logit_scale
        if not torch.isfinite(result).all().item():
            raise JointDecoderError("decoder logits are non-finite")
        return result

    def selected_logits(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        selected_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        if selected_mask.ndim != 2 or tuple(selected_mask.shape) != tuple(mask.shape):
            raise JointDecoderError("selected mask geometry changed")
        selected = selected_mask.to(device=activation.device, dtype=torch.bool)
        if (selected & ~mask).any().item() or selected[:, 0].any().item():
            raise JointDecoderError("selected rows must be valid post-BOS positions")
        indices = torch.nonzero(selected, as_tuple=False)
        if int(indices.shape[0]) <= 0:
            raise JointDecoderError("selected rows are empty")
        hidden = self.projected_hidden(activation, mask)
        return self.logits_from_rows(hidden, indices[:, 0], indices[:, 1], embedding_table)

    def forward(
        self,
        activation: torch.Tensor,
        valid_mask: torch.Tensor,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._check_inputs(activation, valid_mask)
        hidden = self.projected_hidden(activation, mask)
        if tuple(embedding_table.shape) != (self.vocabulary_size, self.hidden_size):
            raise JointDecoderError("embedding table geometry changed")
        logits = hidden.to(embedding_table.dtype) @ embedding_table.transpose(0, 1)
        logits = logits.float() * self.logit_scale
        return torch.where(mask.unsqueeze(-1), logits, torch.zeros_like(logits))

    def affine_only(self) -> "JointAffineAttentionDecoder":
        """Return an affine arm sharing this model's geometry and current state."""

        model = JointAffineAttentionDecoder(
            self.hidden_size,
            self.vocabulary_size,
            AFFINE_METHOD,
            context_width=self.context_width,
        )
        with torch.no_grad():
            model.W.copy_(self.W)
            model.b.copy_(self.b)
            model.s.copy_(self.s)
        return model


def build_decoder(
    method_id: str,
    *,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
    seed: int = DEFAULT_SEED,
    attention_score_mode: str = ATTENTION_SCORE_MODE_DOT_PRODUCT,
) -> JointAffineAttentionDecoder:
    """Build one registered arm with the fixed TRR-0005 initialisation."""

    return JointAffineAttentionDecoder(
        hidden_size,
        vocabulary_size,
        method_id,
        context_width=context_width,
        seed=seed,
        attention_score_mode=attention_score_mode,
    )


@dataclass(frozen=True)
class PositionSchedule:
    """Exact shared draw schedule for one fitting distribution."""

    batch_record_indices: torch.Tensor
    draw_record_slots: torch.Tensor
    draw_position_slots: torch.Tensor
    eligible_counts: torch.Tensor
    used_replacement: torch.Tensor
    seed: int
    position_budget: int
    record_batch_size: int

    @property
    def steps(self) -> int:
        return int(self.batch_record_indices.shape[0])

    @property
    def total_draws(self) -> int:
        return int(self.draw_record_slots.numel())


def build_position_schedule(
    valid_mask: torch.Tensor,
    *,
    steps: int = DEFAULT_STEPS,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    position_budget: int = DEFAULT_POSITION_BUDGET,
    seed: int = DEFAULT_SEED,
) -> PositionSchedule:
    """Build a deterministic 8-record schedule with exactly K post-BOS draws.

    Draws are stored as record-slot/position-slot pairs rather than a boolean
    mask, which preserves exact exposure counts when a small fixture has fewer
    than K eligible positions and sampling with replacement is required.
    """

    if valid_mask.ndim != 2 or valid_mask.shape[0] <= 0 or valid_mask.shape[1] <= 1:
        raise JointDecoderError("valid mask must be [records, positions>1]")
    if valid_mask.dtype not in (torch.bool, torch.uint8):
        raise JointDecoderError("valid mask must be boolean")
    if steps <= 0 or record_batch_size <= 0 or position_budget <= 0:
        raise JointDecoderError("schedule dimensions must be positive")
    mask = valid_mask.to(device="cpu", dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    record_count = int(mask.shape[0])
    batch_indices: list[torch.Tensor] = []
    draw_records: list[torch.Tensor] = []
    draw_positions: list[torch.Tensor] = []
    eligible_counts: list[int] = []
    replacement: list[bool] = []
    for _ in range(int(steps)):
        if record_count >= record_batch_size:
            records = torch.randperm(record_count, generator=generator)[:record_batch_size]
        else:
            records = torch.randint(
                record_count, (record_batch_size,), generator=generator, dtype=torch.long
            )
        eligible = torch.nonzero(mask.index_select(0, records), as_tuple=False)
        eligible = eligible[eligible[:, 1] > 0]
        count = int(eligible.shape[0])
        if count <= 0:
            raise JointDecoderError("schedule batch has no post-BOS eligible position")
        if count >= position_budget:
            chosen = torch.randperm(count, generator=generator)[:position_budget]
            replacement.append(False)
        else:
            chosen = torch.randint(
                count, (position_budget,), generator=generator, dtype=torch.long
            )
            replacement.append(True)
        draws = eligible.index_select(0, chosen)
        batch_indices.append(records)
        draw_records.append(draws[:, 0])
        draw_positions.append(draws[:, 1])
        eligible_counts.append(count)
    return PositionSchedule(
        batch_record_indices=torch.stack(batch_indices).contiguous(),
        draw_record_slots=torch.stack(draw_records).contiguous(),
        draw_position_slots=torch.stack(draw_positions).contiguous(),
        eligible_counts=torch.tensor(eligible_counts, dtype=torch.int32),
        used_replacement=torch.tensor(replacement, dtype=torch.bool),
        seed=int(seed),
        position_budget=int(position_budget),
        record_batch_size=int(record_batch_size),
    )


def schedule_digest(schedule: PositionSchedule) -> str:
    digest = hashlib.sha256(b"trr0005-position-schedule-v1\0")
    for value in (
        schedule.batch_record_indices,
        schedule.draw_record_slots,
        schedule.draw_position_slots,
        schedule.eligible_counts,
        schedule.used_replacement,
    ):
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def checkpoint_steps(steps: int = DEFAULT_STEPS) -> tuple[int, ...]:
    if steps <= 0:
        raise JointDecoderError("steps must be positive")
    values = {0, 25, 50, 75, 100, 150, 200}
    values.update(range(300, int(steps) + 1, 100))
    values.add(int(steps))
    return tuple(sorted(value for value in values if value <= steps))


@dataclass(frozen=True)
class PublicJointData:
    fit_observations: torch.Tensor
    fit_truth: torch.Tensor
    fit_valid_mask: torch.Tensor
    fit_record_ids: tuple[str, ...]
    validation_observations: torch.Tensor
    validation_truth: torch.Tensor
    validation_valid_mask: torch.Tensor
    validation_record_ids: tuple[str, ...]
    validation_groups: tuple[str, ...]
    embedding_table: torch.Tensor
    metadata: Mapping[str, Any]

    @property
    def hidden_size(self) -> int:
        return int(self.fit_observations.shape[-1])

    @property
    def vocabulary_size(self) -> int:
        return int(self.embedding_table.shape[0])


def _json_load(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise JointDecoderError(f"cannot load {label}: {path}") from exc


def _regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise JointDecoderError(f"{label} must be a regular file: {path}")
    return path


def _resource_path(manifest_path: Path, resource: Mapping[str, Any], *, label: str) -> Path:
    raw = resource.get("path")
    if not isinstance(raw, str) or not raw:
        raise JointDecoderError(f"{label} resource has no path")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return _regular_file(path, label=label)


def _load_resource_tensor(
    manifest_path: Path,
    resources: Mapping[str, Any],
    name: str,
    *,
    fallback_names: Sequence[str] = (),
    default_key: str,
    label: str,
) -> tuple[torch.Tensor, Path, str]:
    selected_name = next((candidate for candidate in (name, *fallback_names) if candidate in resources), None)
    if selected_name is None:
        raise JointDecoderError(f"manifest is missing {name} resource")
    resource = resources[selected_name]
    if not isinstance(resource, Mapping):
        raise JointDecoderError(f"{selected_name} resource is malformed")
    path = _resource_path(manifest_path, resource, label=label)
    key = resource.get("tensor_key", default_key)
    if not isinstance(key, str) or not key:
        raise JointDecoderError(f"{selected_name} tensor key is malformed")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in set(handle.keys()):
                raise JointDecoderError(f"{label} is missing tensor {key!r}")
            value = handle.get_tensor(key).contiguous()
    except JointDecoderError:
        raise
    except Exception as exc:
        raise JointDecoderError(f"cannot load {label}: {path}") from exc
    return value, path, key


def _record_manifest(
    manifest_path: Path,
    resources: Mapping[str, Any],
    name: str,
    *,
    fallback_names: Sequence[str] = (),
    label: str,
) -> tuple[list[dict[str, Any]], Path]:
    selected_name = next((candidate for candidate in (name, *fallback_names) if candidate in resources), None)
    if selected_name is None:
        raise JointDecoderError(f"manifest is missing {name} resource")
    resource = resources[selected_name]
    if not isinstance(resource, Mapping):
        raise JointDecoderError(f"{selected_name} resource is malformed")
    path = _resource_path(manifest_path, resource, label=label)
    payload = _json_load(path, label=label)
    values = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not values:
        raise JointDecoderError(f"{label} must contain a non-empty records list")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(values):
        if isinstance(row, str):
            result.append({"record_id": row})
        elif isinstance(row, Mapping) and isinstance(row.get("record_id", row.get("id")), str):
            copied = dict(row)
            copied["record_id"] = str(copied.get("record_id", copied.get("id")))
            result.append(copied)
        else:
            raise JointDecoderError(f"{label} record {index} has no string record_id")
    ids = [str(row["record_id"]) for row in result]
    if len(ids) != len(set(ids)):
        raise JointDecoderError(f"{label} contains duplicate record IDs")
    return result, path


def _record_group(row: Mapping[str, Any]) -> str:
    for key in ("group", "style", "domain", "source"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return "public"


def _validate_observations(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 1 or value.shape[2] <= 0:
        raise JointDecoderError(f"{label} must be [records, positions>1, hidden]")
    if not value.dtype.is_floating_point:
        raise JointDecoderError(f"{label} must be floating point")
    if not torch.isfinite(value).all().item():
        raise JointDecoderError(f"{label} contains non-finite values")
    return value.contiguous()


def _validate_mask(value: torch.Tensor, *, rows: int, positions: int, label: str) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (rows, positions):
        raise JointDecoderError(f"{label} must have shape [{rows}, {positions}]")
    if value.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise JointDecoderError(f"{label} must be boolean or integer")
    result = value.to(device="cpu", dtype=torch.bool).contiguous()
    if not result[:, 0].all().item():
        raise JointDecoderError(f"{label} must contain BOS at every row")
    for row in result:
        false_seen = False
        for item in row.tolist():
            if not item:
                false_seen = True
            elif false_seen:
                raise JointDecoderError(f"{label} must be right-padded")
    return result


def _validate_truth(
    value: torch.Tensor,
    *,
    rows: int,
    positions: int,
    valid_mask: torch.Tensor,
    vocabulary_size: int,
    label: str,
) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (rows, positions):
        raise JointDecoderError(f"{label} must have shape [{rows}, {positions}]")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise JointDecoderError(f"{label} must be integer")
    result = value.to(device="cpu", dtype=torch.long).contiguous()
    if result[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise JointDecoderError(f"{label} rows must begin with BOS token {BOS_TOKEN_ID}")
    scored_mask = valid_mask.clone()
    scored_mask[:, 0] = False
    valid = result[scored_mask]
    if valid.lt(0).any().item() or valid.ge(vocabulary_size).any().item():
        raise JointDecoderError(f"{label} contains an out-of-range token")
    return result


def _load_embedding(
    manifest_path: Path,
    resources: Mapping[str, Any],
    *,
    explicit_path: Path | None = None,
) -> tuple[torch.Tensor, Path, str]:
    if explicit_path is not None:
        path = _regular_file(explicit_path, label="embedding table")
        key = "embeddings"
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if key not in set(handle.keys()):
                    key = "embedding_table" if "embedding_table" in set(handle.keys()) else key
                if key not in set(handle.keys()):
                    raise JointDecoderError("embedding table tensor key is missing")
                value = handle.get_tensor(key).contiguous()
        except JointDecoderError:
            raise
        except Exception as exc:
            raise JointDecoderError(f"cannot load embedding table: {path}") from exc
        return value, path, key
    return _load_resource_tensor(
        manifest_path,
        resources,
        "embedding_table",
        default_key="embeddings",
        label="embedding table",
    )


def _load_split(
    manifest_path: Path,
    *,
    prefix: str,
    vocabulary_size: int | None = None,
    embedding_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one public fit/validation split from a manifest.

    The loader accepts either the TRR-0004 resource names or the compact
    TRR-0005 names.  A combined safetensors artifact is supported by giving
    each logical resource its own ``tensor_key``.
    """

    manifest_path = _regular_file(manifest_path, label=f"{prefix} manifest")
    payload = _json_load(manifest_path, label=f"{prefix} manifest")
    if not isinstance(payload, Mapping):
        raise JointDecoderError(f"{prefix} manifest must be an object")
    schema = payload.get("schema")
    if schema not in (DATA_SCHEMA, "token-reconstruction.trr0005-joint-fit-data.v1", "token-reconstruction.trr0004-public-fit-data.v1"):
        raise JointDecoderError(f"unsupported {prefix} manifest schema: {schema}")
    resources = payload.get("resources")
    if not isinstance(resources, Mapping):
        raise JointDecoderError(f"{prefix} manifest has no resources object")
    obs_name = f"{prefix}_observations"
    truth_name = f"{prefix}_truth"
    mask_name = f"{prefix}_valid_mask"
    records_name = f"{prefix}_records"
    observations, obs_path, obs_key = _load_resource_tensor(
        manifest_path,
        resources,
        obs_name,
        fallback_names=(f"{prefix}_artifact",),
        default_key="activations",
        label=f"{prefix} observations",
    )
    truth, truth_path, truth_key = _load_resource_tensor(
        manifest_path,
        resources,
        truth_name,
        fallback_names=(f"{prefix}_artifact",),
        default_key="token_ids",
        label=f"{prefix} public labels",
    )
    try:
        valid_mask, mask_path, mask_key = _load_resource_tensor(
            manifest_path,
            resources,
            mask_name,
            fallback_names=(f"{prefix}_artifact",),
            default_key="attention_mask",
            label=f"{prefix} validity mask",
        )
    except JointDecoderError as exc:
        if "missing" not in str(exc):
            raise
        valid_mask = torch.ones(tuple(observations.shape[:2]), dtype=torch.bool)
        mask_path, mask_key = obs_path, "implicit_all_valid"
    records, records_path = _record_manifest(
        manifest_path,
        resources,
        records_name,
        label=f"{prefix} record manifest",
    )
    observations = _validate_observations(observations, label=f"{prefix} observations")
    valid_mask = _validate_mask(
        valid_mask,
        rows=int(observations.shape[0]),
        positions=int(observations.shape[1]),
        label=f"{prefix} validity mask",
    )
    vocab = int(vocabulary_size or payload.get("vocabulary_size", VOCAB_SIZE))
    truth = _validate_truth(
        truth,
        rows=int(observations.shape[0]),
        positions=int(observations.shape[1]),
        valid_mask=valid_mask,
        vocabulary_size=vocab,
        label=f"{prefix} public labels",
    )
    ids = tuple(str(row["record_id"]) for row in records)
    if len(ids) != int(observations.shape[0]):
        raise JointDecoderError(f"{prefix} record manifest does not match tensor rows")
    result = {
        "observations": observations,
        "truth": truth,
        "valid_mask": valid_mask,
        "records": records,
        "record_ids": ids,
        "paths": {
            f"{prefix}_observations": {"path": str(obs_path), "key": obs_key},
            f"{prefix}_truth": {"path": str(truth_path), "key": truth_key},
            f"{prefix}_valid_mask": {"path": str(mask_path), "key": mask_key},
            f"{prefix}_records": {"path": str(records_path)},
        },
        "payload": payload,
    }
    if prefix == "validation":
        groups = tuple(_record_group(row) for row in records)
        result["groups"] = groups
    embedding, embedding_file, embedding_key = _load_embedding(
        manifest_path, resources, explicit_path=embedding_path
    )
    if embedding.ndim != 2 or not embedding.dtype.is_floating_point:
        raise JointDecoderError("embedding table must be a floating matrix")
    if int(embedding.shape[1]) != int(observations.shape[2]):
        raise JointDecoderError("embedding and activation hidden sizes differ")
    if vocabulary_size is not None and int(embedding.shape[0]) != int(vocabulary_size):
        raise JointDecoderError("embedding vocabulary size differs from requested size")
    if not torch.isfinite(embedding).all().item():
        raise JointDecoderError("embedding table contains non-finite values")
    result["embedding"] = embedding.contiguous()
    result["paths"]["embedding_table"] = {
        "path": str(embedding_file),
        "key": embedding_key,
    }
    return result, payload


def load_public_joint_data(
    fit_manifest: Path,
    validation_manifest: Path | None = None,
    *,
    embedding_path: Path | None = None,
) -> PublicJointData:
    """Load one distribution and its public selection validation split."""

    fit, fit_payload = _load_split(
        fit_manifest,
        prefix="fit",
        embedding_path=embedding_path,
    )
    validation_source = validation_manifest or fit_manifest
    validation, validation_payload = _load_split(
        validation_source,
        prefix="validation",
        vocabulary_size=int(fit["embedding"].shape[0]),
        embedding_path=embedding_path or Path(fit["paths"]["embedding_table"]["path"]),
    )
    fit_ids = set(fit["record_ids"])
    validation_ids = set(validation["record_ids"])
    if fit_ids.intersection(validation_ids):
        raise JointDecoderError("fit and validation record IDs overlap")
    if tuple(validation["observations"].shape[2:]) != tuple(fit["observations"].shape[2:]):
        raise JointDecoderError("fit and validation hidden geometry differs")
    if not torch.equal(fit["embedding"], validation["embedding"]):
        raise JointDecoderError("fit and validation embedding tables differ")
    metadata = {
        "fit_manifest": str(Path(fit_manifest).expanduser().resolve()),
        "validation_manifest": str(Path(validation_source).expanduser().resolve()),
        "fit_payload": fit_payload,
        "validation_payload": validation_payload,
        "fit_paths": fit["paths"],
        "validation_paths": validation["paths"],
    }
    return PublicJointData(
        fit_observations=fit["observations"],
        fit_truth=fit["truth"],
        fit_valid_mask=fit["valid_mask"],
        fit_record_ids=tuple(fit["record_ids"]),
        validation_observations=validation["observations"],
        validation_truth=validation["truth"],
        validation_valid_mask=validation["valid_mask"],
        validation_record_ids=tuple(validation["record_ids"]),
        validation_groups=tuple(validation["groups"]),
        embedding_table=fit["embedding"],
        metadata=metadata,
    )


def _state_cpu(model: JointAffineAttentionDecoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}


def save_decoder_state(
    path: Path,
    model: JointAffineAttentionDecoder,
    *,
    selected_step: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise JointDecoderError(f"decoder state is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _state_cpu(model)
    state_metadata = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "method_id": model.method_id,
        "selected_step": int(selected_step),
        "attention_mode": model.attention_mode or "none",
        "attention_score_mode": model.attention_score_mode,
        "context_width": int(model.context_width),
        "qkv_init_seed": DEFAULT_SEED,
    }
    if metadata:
        state_metadata.update({str(key): value for key, value in metadata.items()})
    save_file(state, str(path), metadata={key: str(value) for key, value in state_metadata.items()})
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "state_sha256": _tensor_state_digest(state),
        "tensor_sha256": {key: tensor_sha256(value) for key, value in state.items()},
        "state_bytes": sum(int(value.numel()) * value.element_size() for value in state.values()),
        "selected_step": int(selected_step),
        "metadata": state_metadata,
    }


def load_decoder_state(
    path: Path,
    *,
    method_id: str,
    hidden_size: int,
    vocabulary_size: int,
    context_width: int = DEFAULT_CONTEXT_WIDTH,
) -> JointAffineAttentionDecoder:
    """Load a state for final prediction or a public diagnostic."""

    path = _regular_file(path, label="decoder state")
    try:
        state = load_file(str(path), device="cpu")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise JointDecoderError(f"cannot load decoder state: {path}") from exc
    default_score_mode = (
        "none" if method_id == AFFINE_METHOD else ATTENTION_SCORE_MODE_DOT_PRODUCT
    )
    attention_score_mode = metadata.get("attention_score_mode", default_score_mode)
    if not isinstance(attention_score_mode, str):
        raise JointDecoderError("decoder state attention score mode is malformed")
    if method_id == AFFINE_METHOD and attention_score_mode not in ("none", ATTENTION_SCORE_MODE_DOT_PRODUCT):
        raise JointDecoderError("affine decoder state has an attention score mode")
    if method_id != AFFINE_METHOD and attention_score_mode not in ATTENTION_SCORE_MODES:
        raise JointDecoderError(
            f"decoder state attention score mode is unsupported: {attention_score_mode}"
        )
    model = build_decoder(
        method_id,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
        context_width=context_width,
        attention_score_mode=attention_score_mode,
    )
    expected = set(model.state_dict())
    if set(state) != expected:
        raise JointDecoderError(f"decoder state keys differ for {method_id}")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise JointDecoderError(f"decoder state geometry differs for {method_id}") from exc
    return model


def _metric_groups(groups: Sequence[str], totals: Sequence[int], correct: Sequence[int]) -> dict[str, Any]:
    group_total: dict[str, int] = {}
    group_correct: dict[str, int] = {}
    for group, total, hit in zip(groups, totals, correct):
        name = str(group)
        group_total[name] = group_total.get(name, 0) + int(total)
        group_correct[name] = group_correct.get(name, 0) + int(hit)
    if not group_total:
        raise JointDecoderError("validation has no groups")
    accuracies = {name: group_correct[name] / group_total[name] for name in sorted(group_total)}
    total = sum(group_total.values())
    hit = sum(group_correct.values())
    return {
        "token_accuracy": hit / total,
        "correct_tokens": hit,
        "token_rows": total,
        "group_token_accuracy": accuracies,
        "group_token_rows": {name: group_total[name] for name in sorted(group_total)},
        "style_balanced_token_accuracy": sum(accuracies.values()) / len(accuracies),
    }


def evaluate_dataset(
    model: JointAffineAttentionDecoder,
    observations: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    groups: Sequence[str],
    *,
    device: torch.device,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    position_budget: int = DEFAULT_POSITION_BUDGET,
    frequency_reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Evaluate current-position CE in chunks while preserving record metrics."""

    if observations.ndim != 3 or truth.ndim != 2 or valid_mask.ndim != 2:
        raise JointDecoderError("evaluation tensor ranks changed")
    if tuple(truth.shape) != tuple(observations.shape[:2]) or tuple(valid_mask.shape) != tuple(observations.shape[:2]):
        raise JointDecoderError("evaluation tensor geometry differs")
    if len(groups) != int(observations.shape[0]):
        raise JointDecoderError("evaluation groups do not match records")
    if position_budget <= 0 or record_batch_size <= 0:
        raise JointDecoderError("evaluation schedule must be positive")
    started = time.perf_counter()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    projection_calls = 0
    projection_rows = 0
    max_projection_rows = 0
    record_totals: list[int] = [0 for _ in range(int(observations.shape[0]))]
    record_correct: list[int] = [0 for _ in range(int(observations.shape[0]))]
    frequency_counts: torch.Tensor | None = None
    if frequency_reference is not None:
        reference = frequency_reference.reshape(-1).to(device="cpu", dtype=torch.long)
        if reference.numel() <= 0:
            raise JointDecoderError("frequency reference is empty")
        if reference.lt(0).any().item() or reference.ge(model.vocabulary_size).any().item():
            raise JointDecoderError("frequency reference contains an out-of-range token")
        frequency_counts = torch.bincount(reference, minlength=model.vocabulary_size)
    frequency_bucket_rows: dict[str, int] = {}
    frequency_bucket_correct: dict[str, int] = {}
    position_bucket_rows: dict[str, int] = {}
    position_bucket_correct: dict[str, int] = {}
    frequency_position_rows: dict[str, dict[str, int]] = {}
    frequency_position_correct: dict[str, dict[str, int]] = {}
    runtime_embedding = embedding_table.to(device=device)
    with torch.inference_mode():
        for start in range(0, int(observations.shape[0]), record_batch_size):
            stop = min(start + record_batch_size, int(observations.shape[0]))
            activation = observations[start:stop].to(device=device, dtype=torch.float32)
            mask = valid_mask[start:stop].to(device=device, dtype=torch.bool)
            labels = truth[start:stop].to(device=device, dtype=torch.long)
            hidden = model.projected_hidden(activation, mask)
            indices = torch.nonzero(mask, as_tuple=False)
            indices = indices[indices[:, 1] > 0]
            for chunk in indices.split(position_budget):
                if int(chunk.shape[0]) <= 0:
                    continue
                rows = model.logits_from_rows(hidden, chunk[:, 0], chunk[:, 1], runtime_embedding)
                target = labels[chunk[:, 0], chunk[:, 1]]
                loss = F.cross_entropy(rows, target, reduction="sum")
                prediction = rows.argmax(dim=-1)
                hits = prediction.eq(target)
                # Keep metric bookkeeping off the hot device path: transfer each
                # completed chunk vector once, then count records and histograms on CPU.
                indices_cpu = chunk.detach().cpu()
                prediction_cpu = prediction.detach().cpu()
                target_cpu = target.detach().cpu()
                hits_cpu = hits.detach().cpu()
                total_loss += float(loss.detach().cpu())
                total_correct += int(hits_cpu.sum())
                total_rows += int(target.numel())
                projection_calls += 1
                projection_rows += int(target.numel())
                max_projection_rows = max(max_projection_rows, int(target.numel()))
                for local_index in range(int(chunk.shape[0])):
                    record = int(indices_cpu[local_index, 0]) + start
                    record_totals[record] += 1
                    record_correct[record] += int(prediction_cpu[local_index] == target_cpu[local_index])
                if frequency_counts is not None:
                    for local_index in range(int(chunk.shape[0])):
                        target_id = int(target_cpu[local_index])
                        frequency = int(frequency_counts[target_id].item())
                        frequency_bucket = (
                            "unseen_0" if frequency == 0
                            else "seen_1_4" if frequency <= 4
                            else "seen_5_19" if frequency <= 19
                            else "seen_20_plus"
                        )
                        position = int(indices_cpu[local_index, 1])
                        position_bucket = (
                            "1-15" if position <= 15
                            else "16-39" if position <= 39
                            else "40-79" if position <= 79
                            else "80+"
                        )
                        hit = int(hits_cpu[local_index])
                        frequency_bucket_rows[frequency_bucket] = frequency_bucket_rows.get(frequency_bucket, 0) + 1
                        frequency_bucket_correct[frequency_bucket] = frequency_bucket_correct.get(frequency_bucket, 0) + hit
                        position_bucket_rows[position_bucket] = position_bucket_rows.get(position_bucket, 0) + 1
                        position_bucket_correct[position_bucket] = position_bucket_correct.get(position_bucket, 0) + hit
                        row_counts = frequency_position_rows.setdefault(frequency_bucket, {})
                        row_correct = frequency_position_correct.setdefault(frequency_bucket, {})
                        row_counts[position_bucket] = row_counts.get(position_bucket, 0) + 1
                        row_correct[position_bucket] = row_correct.get(position_bucket, 0) + hit
    if total_rows <= 0:
        raise JointDecoderError("evaluation has no post-BOS rows")
    metrics = _metric_groups(groups, record_totals, record_correct)
    result: dict[str, Any] = {
        "loss": total_loss / total_rows,
        **metrics,
        "exact_records": sum(
            int(total > 0 and total == correct)
            for total, correct in zip(record_totals, record_correct)
        ),
        "record_count": len(record_totals),
        "record_correct_tokens": record_correct,
        "record_token_rows": record_totals,
        "projection_calls": projection_calls,
        "projection_rows": projection_rows,
        "max_projection_rows": max_projection_rows,
        "evaluation_seconds": time.perf_counter() - started,
    }
    if frequency_counts is not None:
        ordered_frequency = ("unseen_0", "seen_1_4", "seen_5_19", "seen_20_plus")
        ordered_position = ("1-15", "16-39", "40-79", "80+")
        result["frequency_reference_token_rows"] = int(frequency_reference.numel())
        result["frequency_bucket_metrics"] = {
            key: {
                "rows": int(frequency_bucket_rows.get(key, 0)),
                "correct": int(frequency_bucket_correct.get(key, 0)),
                "token_accuracy": (
                    frequency_bucket_correct.get(key, 0) / frequency_bucket_rows[key]
                    if frequency_bucket_rows.get(key, 0) else None
                ),
            }
            for key in ordered_frequency
        }
        result["position_bucket_metrics"] = {
            key: {
                "rows": int(position_bucket_rows.get(key, 0)),
                "correct": int(position_bucket_correct.get(key, 0)),
                "token_accuracy": (
                    position_bucket_correct.get(key, 0) / position_bucket_rows[key]
                    if position_bucket_rows.get(key, 0) else None
                ),
            }
            for key in ordered_position
        }
        result["frequency_by_position_bucket"] = {
            frequency: {
                position: {
                    "rows": int(frequency_position_rows.get(frequency, {}).get(position, 0)),
                    "correct": int(frequency_position_correct.get(frequency, {}).get(position, 0)),
                    "token_accuracy": (
                        frequency_position_correct.get(frequency, {}).get(position, 0)
                        / frequency_position_rows[frequency][position]
                        if frequency_position_rows.get(frequency, {}).get(position, 0) else None
                    ),
                }
                for position in ordered_position
            }
            for frequency in ordered_frequency
        }
    return result

def train_step(
    model: JointAffineAttentionDecoder,
    observations: torch.Tensor,
    truth: torch.Tensor,
    valid_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    schedule: PositionSchedule,
    step_index: int,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
) -> dict[str, Any]:
    """Run one exact-K current-position CE update."""

    model.train()
    batch_indices = schedule.batch_record_indices[step_index]
    activation = observations.index_select(0, batch_indices).to(device=device, dtype=torch.float32)
    mask = valid_mask.index_select(0, batch_indices).to(device=device, dtype=torch.bool)
    labels = truth.index_select(0, batch_indices).to(device=device, dtype=torch.long)
    record_slots = schedule.draw_record_slots[step_index].to(device=device)
    position_slots = schedule.draw_position_slots[step_index].to(device=device)
    if int(record_slots.numel()) != schedule.position_budget:
        raise JointDecoderError("schedule draw count changed")
    if position_slots.eq(0).any().item() or (~mask[record_slots, position_slots]).any().item():
        raise JointDecoderError("schedule contains an invalid or BOS draw")
    hidden = model.projected_hidden(activation, mask)
    logits = model.logits_from_rows(hidden, record_slots, position_slots, embedding_table.to(device=device))
    target = labels[record_slots, position_slots]
    loss = F.cross_entropy(logits, target)
    if not torch.isfinite(loss).item():
        raise JointDecoderError("training loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(model.parameters()), gradient_clip_norm, error_if_nonfinite=True
    )
    gradient_norms = {
        name: (None if parameter.grad is None else float(parameter.grad.detach().norm().cpu()))
        for name, parameter in model.named_parameters()
    }
    optimizer.step()
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all().item():
            raise JointDecoderError("model parameter became non-finite")
    return {
        "loss": float(loss.detach().cpu()),
        "correct_tokens": int(logits.detach().argmax(dim=-1).eq(target).sum().cpu()),
        "token_rows": int(target.numel()),
        "token_accuracy": float(logits.detach().argmax(dim=-1).eq(target).float().mean().cpu()),
        "gradient_norm": float(grad_norm.detach().cpu()),
        "gradient_norms": gradient_norms,
        "q_gradient_norm": gradient_norms.get("query.weight") if model.query is not None else None,
        "k_gradient_norm": gradient_norms.get("key.weight") if model.key is not None else None,
        "v_gradient_norm": gradient_norms.get("value.weight") if model.value is not None else None,
        "output_gradient_norm": gradient_norms.get("output.weight") if model.output is not None else None,
        "draws": int(target.numel()),
        "used_replacement": bool(schedule.used_replacement[step_index].item()),
        "eligible_positions": int(schedule.eligible_counts[step_index].item()),
    }


def schedule_metadata(schedule: PositionSchedule) -> dict[str, Any]:
    # Replacement is only permitted within a step when that batch has fewer
    # than K eligible positions.  Count unique/repeated pairs within each
    # step so the receipt distinguishes intentional replacement from ordinary
    # repeated exposure across the 3000-step schedule.
    position_base = max(1, int(schedule.draw_position_slots.max().item()) + 1)
    within_step_unique = 0
    replacement_unique = 0
    replacement_repeated = 0
    for step_index in range(schedule.steps):
        if bool(schedule.used_replacement[step_index].item()):
            encoded = (
                schedule.draw_record_slots[step_index].to(torch.int64) * position_base
                + schedule.draw_position_slots[step_index].to(torch.int64)
            )
            unique = int(torch.unique(encoded).numel())
            replacement_unique += unique
            replacement_repeated += int(schedule.position_budget) - unique
        else:
            unique = int(schedule.position_budget)
        within_step_unique += unique
    total_draws = int(schedule.total_draws)
    within_step_repeated = total_draws - within_step_unique
    return {
        "seed": int(schedule.seed),
        "steps": schedule.steps,
        "record_batch_size": int(schedule.record_batch_size),
        "position_budget": int(schedule.position_budget),
        "total_draws": total_draws,
        "expected_total_draws": int(schedule.steps * schedule.position_budget),
        "unique_draws_within_step": within_step_unique,
        "repeated_draws_within_step": within_step_repeated,
        "replacement_unique_draws": replacement_unique,
        "replacement_repeated_draws": replacement_repeated,
        "eligible_min": int(schedule.eligible_counts.min().item()),
        "eligible_max": int(schedule.eligible_counts.max().item()),
        "replacement_steps": int(schedule.used_replacement.sum().item()),
        "replacement_required": bool(schedule.used_replacement.any().item()),
        "schedule_sha256": schedule_digest(schedule),
    }


def save_schedule(path: Path, schedule: PositionSchedule) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise JointDecoderError(f"position schedule is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "batch_record_indices": schedule.batch_record_indices.to(dtype=torch.int32),
        "draw_record_slots": schedule.draw_record_slots.to(dtype=torch.int16),
        "draw_position_slots": schedule.draw_position_slots.to(dtype=torch.int16),
        "eligible_counts": schedule.eligible_counts,
        "used_replacement": schedule.used_replacement.to(dtype=torch.uint8),
    }
    save_file(tensors, str(path), metadata={key: str(value) for key, value in schedule_metadata(schedule).items()})
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        **schedule_metadata(schedule),
    }

