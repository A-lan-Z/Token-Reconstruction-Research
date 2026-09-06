"""Activation-only causal students for TRR-P04.

The P04 student receives only a boundary-activation prefix for each output
position.  Its retained output table is the fixed public normalized embedding
table.  The affine path is trainable in every arm; the GRU residual is
zero-initialized so the recurrent students begin at the same affine function.
No candidate set, teacher score, source token, public prefix, or guessed-token
feedback is represented by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn


TASK_ID = "TRR-P04"
STUDENT_SCHEMA = "token-reconstruction.trr-p04-student.v1"
STATE_SCHEMA = "token-reconstruction.trr-p04-student-state.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr-p04-predictions.v1"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
DEFAULT_GRU_WIDTH = 256
DEFAULT_LOGIT_SCALE = 3.0
METHOD_AFFINE = "affine_same_data"
METHOD_S = "student_s"
METHOD_H = "student_h"
METHOD_D = "student_d"
STUDENT_METHODS = (METHOD_S, METHOD_H, METHOD_D)
ALL_METHODS = (METHOD_AFFINE, *STUDENT_METHODS)


class P04StudentError(RuntimeError):
    """Raised when a P04 student violates its activation-only contract."""


@dataclass(frozen=True)
class StudentArchitectureConfig:
    """Architecture constants shared by all P04 arms."""

    hidden_size: int = HIDDEN_SIZE
    vocab_size: int = VOCAB_SIZE
    gru_width: int = DEFAULT_GRU_WIDTH
    initial_logit_scale: float = DEFAULT_LOGIT_SCALE
    input_normalization: str = (
        "F.layer_norm(x, (hidden_size,), weight=None, bias=None, eps=1e-5)"
    )
    output_normalization: str = "F.normalize(affine(x) + gru_up(GRU(layer_norm(x))), dim=-1)"
    output_table: str = "fixed normalized public input-embedding table"

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.vocab_size <= 0 or self.gru_width <= 0:
            raise P04StudentError("student geometry must be positive")
        if not torch.isfinite(torch.tensor(self.initial_logit_scale)):
            raise P04StudentError("student initial logit scale must be finite")
        if self.initial_logit_scale <= 0:
            raise P04StudentError("student initial logit scale must be positive")


class StudentModel(Protocol):
    """Structural protocol for the shared student prediction helpers."""

    hidden_size: int
    vocab_size: int

    def projected_hidden(self, activation: torch.Tensor) -> torch.Tensor: ...

    @property
    def logit_scale(self) -> torch.Tensor: ...



def _check_activation(activation: torch.Tensor, *, hidden_size: int) -> None:
    if activation.ndim not in (2, 3) or activation.shape[-1] != hidden_size:
        raise P04StudentError("activation must end in the declared hidden size")
    if not activation.dtype.is_floating_point:
        raise P04StudentError("activation must be floating point")
    if not torch.isfinite(activation).all().item():
        raise P04StudentError("activation contains non-finite values")


def validate_embedding_table(
    embedding_table: torch.Tensor, *, hidden_size: int, vocab_size: int, require_unit_norm: bool = True
) -> None:
    if embedding_table.ndim != 2 or tuple(embedding_table.shape) != (vocab_size, hidden_size):
        raise P04StudentError("embedding table geometry changed")
    if not embedding_table.dtype.is_floating_point:
        raise P04StudentError("embedding table must be floating point")
    if not torch.isfinite(embedding_table).all().item():
        raise P04StudentError("embedding table contains non-finite values")
    if require_unit_norm:
        norms = torch.linalg.vector_norm(embedding_table.float(), dim=-1)
        allowed = torch.isclose(norms, torch.ones_like(norms), atol=2e-4, rtol=2e-4)
        allowed |= torch.isclose(norms, torch.zeros_like(norms), atol=2e-6, rtol=0.0)
        if not allowed.all().item():
            raise P04StudentError("embedding table is not normalized")


def normalize_public_embeddings(embedding_table: torch.Tensor) -> torch.Tensor:
    if embedding_table.ndim != 2 or not embedding_table.dtype.is_floating_point:
        raise P04StudentError("public embedding table must be a floating-point matrix")
    if not torch.isfinite(embedding_table).all().item():
        raise P04StudentError("public embedding table is non-finite")
    result = F.normalize(embedding_table.detach().float(), dim=-1)
    if not torch.isfinite(result).all().item():
        raise P04StudentError("normalized public embedding table is non-finite")
    return result.contiguous()


def _deterministic_lowest_id(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-1 IDs and exact tie counts with lowest-ID ties.

    PyTorch's ``argmax`` returns the first maximal index, so the vocabulary
    dimension's ascending token-ID order implements the declared lowest-ID
    tie rule without constructing a vocabulary-sized cumulative mask.
    """

    if logits.ndim < 2 or logits.shape[-1] <= 0:
        raise P04StudentError("logits must have a non-empty vocabulary dimension")
    if not torch.isfinite(logits).all().item():
        raise P04StudentError("logits contain non-finite values")
    maxima = logits.amax(dim=-1, keepdim=True)
    tie_count = logits.eq(maxima).sum(dim=-1).to(dtype=torch.int32)
    ids = logits.argmax(dim=-1).to(dtype=torch.int64)
    return ids, tie_count


def _check_embedding_table_geometry(
    embedding_table: torch.Tensor, *, hidden_size: int, vocab_size: int
) -> None:
    """Check cheap table invariants in projection hot paths.

    Unit-norm and finite-value scans happen once at asset load or prediction
    setup. Per-update projection only repeats shape/device/dtype checks.
    """

    if embedding_table.ndim != 2 or tuple(embedding_table.shape) != (vocab_size, hidden_size):
        raise P04StudentError("embedding table geometry changed")
    if not embedding_table.dtype.is_floating_point:
        raise P04StudentError("embedding table must be floating point")


class _AffinePath(nn.Module):
    """Trainable full hidden-space affine path shared by every arm."""

    def __init__(
        self,
        hidden_size: int,
        *,
        initial_logit_scale: float,
        initial_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_size, dtype=torch.float32))
            self.linear.bias.zero_()
        self.log_scale = nn.Parameter(torch.tensor(float(initial_logit_scale), dtype=torch.float32))
        if initial_state is not None:
            required = {"W", "b", "s"}
            if set(initial_state) != required:
                raise P04StudentError("affine initialization must contain exactly W, b, and s")
            W = initial_state["W"].detach().cpu().float()
            b = initial_state["b"].detach().cpu().float()
            s = initial_state["s"].detach().cpu().float()
            if W.shape != (hidden_size, hidden_size) or b.shape != (hidden_size,) or s.ndim != 0:
                raise P04StudentError("affine initialization geometry changed")
            if not all(torch.isfinite(value).all().item() for value in (W, b, s)):
                raise P04StudentError("affine initialization contains non-finite values")
            with torch.no_grad():
                self.linear.weight.copy_(W)
                self.linear.bias.copy_(b)
                self.log_scale.copy_(s)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.linear(activation.float())

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.log_scale


class AffineStudent(nn.Module):
    """Same-data affine comparator with the P04 tied public classifier."""

    method_id = METHOD_AFFINE
    uses_gru = False

    def __init__(
        self,
        config: StudentArchitectureConfig | None = None,
        *,
        affine_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or StudentArchitectureConfig()
        self.config.validate()
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.affine = _AffinePath(
            self.hidden_size,
            initial_logit_scale=self.config.initial_logit_scale,
            initial_state=affine_state,
        )

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.affine.logit_scale

    def projected_hidden(self, activation: torch.Tensor) -> torch.Tensor:
        _check_activation(activation, hidden_size=self.hidden_size)
        return F.normalize(self.affine(activation), dim=-1)

    def selected_logits(
        self, activation: torch.Tensor, selected_mask: torch.Tensor, embedding_table: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.projected_hidden(activation)
        return _selected_logits(hidden, selected_mask, embedding_table, self.logit_scale, self.vocab_size)


class GRUAffineStudent(nn.Module):
    """One-layer unidirectional GRU residual over a trainable affine path.

    The GRU sees the activation sequence after fixed layer normalization.  Its
    output projection is zero-initialized, making step zero exactly the shared
    affine comparator while retaining a trainable causal path.  A call starts
    with no hidden state, so recurrent state is reset for each record batch.
    """

    method_id = METHOD_S
    uses_gru = True

    def __init__(
        self,
        config: StudentArchitectureConfig | None = None,
        *,
        affine_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or StudentArchitectureConfig()
        self.config.validate()
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.gru_width = self.config.gru_width
        self.affine = _AffinePath(
            self.hidden_size,
            initial_logit_scale=self.config.initial_logit_scale,
            initial_state=affine_state,
        )
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.gru_width,
            num_layers=1,
            batch_first=True,
        )
        self.gru_up = nn.Linear(self.gru_width, self.hidden_size, bias=True)
        with torch.no_grad():
            self.gru_up.weight.zero_()
            self.gru_up.bias.zero_()

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.affine.logit_scale

    def pre_normalized_hidden(self, activation: torch.Tensor) -> torch.Tensor:
        _check_activation(activation, hidden_size=self.hidden_size)
        if activation.ndim != 3:
            raise P04StudentError("GRU sequence input must be [batch, time, hidden]")
        value = activation.float()
        normalized = F.layer_norm(
            value,
            (self.hidden_size,),
            weight=None,
            bias=None,
            eps=1e-5,
        )
        recurrent, _ = self.gru(normalized)
        return self.affine(value) + self.gru_up(recurrent)

    def projected_hidden(self, activation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.pre_normalized_hidden(activation), dim=-1)

    def selected_logits(
        self, activation: torch.Tensor, selected_mask: torch.Tensor, embedding_table: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.projected_hidden(activation)
        return _selected_logits(hidden, selected_mask, embedding_table, self.logit_scale, self.vocab_size)


def _selected_logits(
    projected_hidden: torch.Tensor,
    selected_mask: torch.Tensor,
    embedding_table: torch.Tensor,
    log_scale: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    if projected_hidden.ndim != 3 or selected_mask.ndim != 2:
        raise P04StudentError("selected projection expects sequence hidden and mask")
    if tuple(projected_hidden.shape[:2]) != tuple(selected_mask.shape):
        raise P04StudentError("selected mask geometry changed")
    if selected_mask.dtype not in (torch.bool, torch.uint8):
        raise P04StudentError("selected mask must be boolean")
    _check_embedding_table_geometry(
        embedding_table,
        hidden_size=int(projected_hidden.shape[-1]),
        vocab_size=vocab_size,
    )
    selected = selected_mask.to(device=projected_hidden.device, dtype=torch.bool)
    if not selected.any().item():
        raise P04StudentError("selected mask contains no active positions")
    hidden = projected_hidden[selected]
    table = embedding_table.to(device=projected_hidden.device, dtype=torch.float32)
    scale = log_scale.float().exp()
    if not torch.isfinite(scale).item() or scale.item() <= 0:
        raise P04StudentError("student logit scale is invalid")
    logits = hidden @ table.transpose(0, 1)
    logits = logits * scale
    if not torch.isfinite(logits).all().item():
        raise P04StudentError("student logits are non-finite")
    return logits


def build_student(
    method_id: str,
    *,
    config: StudentArchitectureConfig | None = None,
    affine_state: Mapping[str, torch.Tensor] | None = None,
) -> nn.Module:
    if method_id == METHOD_AFFINE:
        return AffineStudent(config, affine_state=affine_state)
    if method_id in STUDENT_METHODS:
        model = GRUAffineStudent(config, affine_state=affine_state)
        model.method_id = method_id
        return model
    raise P04StudentError(f"unknown P04 student method: {method_id}")


def initialize_student(
    method_id: str,
    *,
    seed: int,
    config: StudentArchitectureConfig | None = None,
    affine_state: Mapping[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Initialize one arm reproducibly; paired GRU arms share exact state."""

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return build_student(method_id, config=config, affine_state=affine_state)


def prediction_tensor(
    model: StudentModel,
    activations: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    valid_mask: torch.Tensor | None = None,
    record_batch_size: int = 8,
    projection_chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict active positions with unrestricted full-vocabulary logits.

    ``valid_mask`` is optional for fixed-length inputs. When supplied it must
    be a boolean ``[records, time]`` mask with right-padded inactive columns;
    inactive outputs remain ``-1`` with a zero tie count. The mask is a
    deployment-time observation mask only and never changes the vocabulary
    decision rule.
    """

    if activations.ndim != 3 or activations.shape[1] <= 1:
        raise P04StudentError("prediction activations must be [records,time>1,hidden]")
    if record_batch_size <= 0 or projection_chunk <= 0:
        raise P04StudentError("prediction batching must be positive")
    validate_embedding_table(
        embedding_table,
        hidden_size=int(activations.shape[-1]),
        vocab_size=int(model.vocab_size),
        require_unit_norm=True,
    )
    model.eval()
    rows, sequence, _ = map(int, activations.shape)
    predictions = torch.full((rows, sequence), -1, dtype=torch.int32)
    tie_counts = torch.zeros((rows, sequence), dtype=torch.int32)
    if valid_mask is None:
        valid = torch.ones((rows, sequence), dtype=torch.bool)
    else:
        if valid_mask.shape != (rows, sequence) or valid_mask.dtype is not torch.bool:
            raise P04StudentError("valid mask must be boolean [records,time]")
        valid = valid_mask.detach().cpu().contiguous()
        if not torch.equal(valid, valid.cumprod(dim=1).to(torch.bool)):
            raise P04StudentError("valid mask must be right-padded per record")
        if not valid.any(dim=1).all().item():
            raise P04StudentError("each record must have at least one active position")
    table = embedding_table.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, rows, record_batch_size):
            stop = min(start + record_batch_size, rows)
            batch = activations[start:stop].to(device=device, dtype=torch.float32)
            projected = model.projected_hidden(batch)
            flat = projected.reshape(-1, int(projected.shape[-1]))
            batch_valid = valid[start:stop].reshape(-1).to(device=device)
            active_indices = torch.nonzero(batch_valid, as_tuple=False).reshape(-1)
            active = flat.index_select(0, active_indices)
            for chunk_start in range(0, active.shape[0], projection_chunk):
                chunk_stop = min(chunk_start + projection_chunk, int(active.shape[0]))
                logits = active[chunk_start:chunk_stop] @ table.transpose(0, 1)
                scale = model.logit_scale.float().exp()
                if not torch.isfinite(scale).item() or scale.item() <= 0:
                    raise P04StudentError("student logit scale is invalid")
                logits = logits * scale
                ids, ties = _deterministic_lowest_id(logits)
                destination = active_indices[chunk_start:chunk_stop].to(device="cpu")
                flat_predictions = predictions[start:stop].reshape(-1)
                flat_ties = tie_counts[start:stop].reshape(-1)
                flat_predictions[destination] = ids.to(device="cpu", dtype=torch.int32)
                flat_ties[destination] = ties.to(device="cpu", dtype=torch.int32)
    return predictions, tie_counts


def state_tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"trr-p04-student-state-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_student_state(
    model: nn.Module,
    path: Path,
    *,
    method_id: str,
    seed: int,
    config: StudentArchitectureConfig,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise P04StudentError(f"student state is create-only: {path}")
    if method_id not in ALL_METHODS:
        raise P04StudentError("cannot save an unknown P04 method")
    config.validate()
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(path),
        metadata={
            "schema": STATE_SCHEMA,
            "task_id": TASK_ID,
            "method_id": method_id,
            "seed": str(seed),
            "architecture_json": json.dumps(asdict(config), sort_keys=True),
            **{str(key): str(value) for key, value in (metadata or {}).items()},
        },
    )
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": state_sha256(path),
        "state_sha256": state_tensor_digest(state),
        "state_bytes": sum(int(value.numel()) * value.element_size() for value in state.values()),
        "tensor_sha256": {key: state_tensor_digest({key: value}) for key, value in state.items()},
        "method_id": method_id,
        "seed": int(seed),
    }


def _metadata(path: Path) -> dict[str, str]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            value = handle.metadata()
    except Exception as exc:  # pragma: no cover - backend-specific
        raise P04StudentError(f"cannot read student-state metadata: {path}") from exc
    return dict(value or {})


def load_student_state(
    path: Path,
    *,
    method_id: str,
    device: torch.device,
    config: StudentArchitectureConfig | None = None,
) -> nn.Module:
    if path.is_symlink() or not path.is_file():
        raise P04StudentError(f"student state must be a regular file: {path}")
    metadata = _metadata(path)
    if metadata.get("schema") != STATE_SCHEMA or metadata.get("method_id") != method_id:
        raise P04StudentError("student state schema or method binding changed")
    state = load_file(str(path), device="cpu")
    model = build_student(method_id, config=config)
    if set(state) != set(model.state_dict()):
        raise P04StudentError("student state tensor fields changed")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def model_parameter_summary(model: nn.Module) -> dict[str, int]:
    return {
        "trainable_parameters": sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
        "all_parameters": sum(int(parameter.numel()) for parameter in model.parameters()),
        "trainable_state_bytes": sum(
            int(parameter.numel()) * parameter.element_size()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def candidate_free_inference_contract() -> dict[str, Any]:
    return {
        "input": "activation prefix H[0:i+1] represented by one causal sequence pass",
        "uses_source_tokens": False,
        "uses_teacher_tokens_or_scores": False,
        "uses_candidate_ids": False,
        "uses_public_prefix": False,
        "uses_a2_or_search": False,
        "feedback": False,
        "output": "unrestricted full-vocabulary tied-embedding logits and lowest-ID top-1",
        "record_reset": True,
    }


__all__ = [
    "ALL_METHODS",
    "AffineStudent",
    "BOS_TOKEN_ID",
    "DEFAULT_GRU_WIDTH",
    "GRUAffineStudent",
    "METHOD_AFFINE",
    "METHOD_D",
    "METHOD_H",
    "METHOD_S",
    "P04StudentError",
    "PREDICTION_SCHEMA",
    "STUDENT_METHODS",
    "StudentArchitectureConfig",
    "build_student",
    "candidate_free_inference_contract",
    "initialize_student",
    "load_student_state",
    "model_parameter_summary",
    "normalize_public_embeddings",
    "prediction_tensor",
    "save_student_state",
    "state_sha256",
    "state_tensor_digest",
    "validate_embedding_table",
]
