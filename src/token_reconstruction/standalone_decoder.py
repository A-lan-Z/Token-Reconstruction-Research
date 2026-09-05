"""Standalone token decoders for the TRR-0003 Track B pilot.

The deployed models in this module make one direct token prediction for every
observed position.  They do not generate candidate sets and do not call the
public prefix at inference time.  The public input-embedding table is a fixed
runtime resource; the learned state is kept separate so its retained size is
visible in experiment evidence.

The affine arm is deliberately tied to the public embedding table.  This
avoids comparing a 2048 x 128256 untied classifier (roughly 1 GiB of weights)
against a compact inverse while still training with a full-vocabulary token
classification objective.  The angular arm is the existing residual affine
inverse and is included as a loss/objective control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn

from .inverse import InverseTrainingConfig, ResidualAffineInverse, train_inverse


TRACK_B_SCHEMA = "token-reconstruction.trr0003-track-b.v1"
AFFINE_SCHEMA = "token-reconstruction.trr0003-tied-affine-ce.v1"
MLP_SCHEMA = "token-reconstruction.trr0003-residual-mlp-ce.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0003-standalone-predictions.v1"


class StandaloneDecoderError(RuntimeError):
    """Raised when a Track B decoder or artifact violates its contract."""


@dataclass(frozen=True)
class DecoderTrainingConfig:
    """Fixed optimizer settings for one standalone decoder fit."""

    steps: int = 600
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    log_every: int = 25
    logit_scale: float = 16.0
    seed: int = 1737

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.log_every <= 0:
            raise StandaloneDecoderError("decoder training schedule must be positive")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise StandaloneDecoderError("decoder optimizer settings must be positive")
        if self.weight_decay < 0 or self.logit_scale <= 0:
            raise StandaloneDecoderError("decoder optimizer settings are invalid")


def _check_matrix(value: torch.Tensor, *, name: str, width: int | None = None) -> None:
    if value.ndim != 2:
        raise StandaloneDecoderError(f"{name} must be a matrix")
    if value.shape[0] <= 0 or (width is not None and value.shape[1] != width):
        raise StandaloneDecoderError(f"{name} has invalid geometry")
    if not value.dtype.is_floating_point:
        raise StandaloneDecoderError(f"{name} must be floating point")
    if not torch.isfinite(value).all().item():
        raise StandaloneDecoderError(f"{name} contains non-finite values")


def validate_training_tensors(
    activations: torch.Tensor,
    labels: torch.Tensor,
    embedding_table: torch.Tensor,
) -> None:
    """Validate public fitting tensors before any optimizer state is created."""

    _check_matrix(activations, name="activations")
    if labels.ndim != 1 or labels.shape[0] != activations.shape[0]:
        raise StandaloneDecoderError("labels must match activation rows")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise StandaloneDecoderError("labels must be integer token IDs")
    if labels.lt(0).any().item() or labels.ge(embedding_table.shape[0]).any().item():
        raise StandaloneDecoderError("labels contain an out-of-range token ID")
    _check_matrix(embedding_table, name="embedding_table", width=int(activations.shape[1]))


def normalized_embedding_table(embedding_table: torch.Tensor) -> torch.Tensor:
    """Return a checked, detached float32 table used by all tied decoders."""

    _check_matrix(embedding_table, name="embedding_table")
    result = F.normalize(embedding_table.detach().float(), dim=-1)
    if not torch.isfinite(result).all().item():
        raise StandaloneDecoderError("normalized embedding table is non-finite")
    return result.contiguous()


class TiedAffineTokenDecoder(nn.Module):
    """Identity-plus-affine map with a tied public embedding classifier.

    The residual starts at zero, matching the existing inverse's structural
    initialization.  A trainable vocabulary bias is the only classifier state;
    the large embedding table remains a fixed external public resource.
    """

    method_id = "tied_affine_token_ce"

    def __init__(self, hidden_size: int, vocab_size: int, *, logit_scale: float = 16.0) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0 or logit_scale <= 0:
            raise StandaloneDecoderError("invalid affine decoder geometry")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.logit_scale = float(logit_scale)
        self.residual = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.classifier_bias = nn.Parameter(torch.zeros(self.vocab_size))

    def transformed(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.shape[-1] != self.hidden_size:
            raise StandaloneDecoderError("activation hidden size changed")
        value = activation.float()
        return F.normalize(value + self.residual(value), dim=-1)

    def forward(self, activation: torch.Tensor, embedding_table: torch.Tensor) -> torch.Tensor:
        embeddings = _checked_runtime_embeddings(
            embedding_table,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
        )
        return (
            self.transformed(activation) @ embeddings.transpose(0, 1) * self.logit_scale
            + self.classifier_bias
        )


class ResidualMLPTokenDecoder(nn.Module):
    """Compact nonlinear residual decoder with a tied public classifier."""

    method_id = "residual_mlp256_token_ce"

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        bottleneck_size: int = 256,
        logit_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if min(hidden_size, vocab_size, bottleneck_size) <= 0 or logit_scale <= 0:
            raise StandaloneDecoderError("invalid residual MLP geometry")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.bottleneck_size = int(bottleneck_size)
        self.logit_scale = float(logit_scale)
        self.down = nn.Linear(self.hidden_size, self.bottleneck_size)
        self.up = nn.Linear(self.bottleneck_size, self.hidden_size)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        # A small nonzero output keeps the first update informative while the
        # residual remains close to the identity at initialization.
        nn.init.normal_(self.up.weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.up.bias)
        self.classifier_bias = nn.Parameter(torch.zeros(self.vocab_size))

    def transformed(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.shape[-1] != self.hidden_size:
            raise StandaloneDecoderError("activation hidden size changed")
        value = activation.float()
        return F.normalize(value + self.up(F.gelu(self.down(value))), dim=-1)

    def forward(self, activation: torch.Tensor, embedding_table: torch.Tensor) -> torch.Tensor:
        embeddings = _checked_runtime_embeddings(
            embedding_table,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
        )
        return (
            self.transformed(activation) @ embeddings.transpose(0, 1) * self.logit_scale
            + self.classifier_bias
        )


def validate_embedding_table(embedding_table: torch.Tensor, *, hidden_size: int, vocab_size: int) -> None:
    """Validate a loaded public table once before entering a hot inference loop."""

    if embedding_table.ndim != 2 or tuple(embedding_table.shape) != (vocab_size, hidden_size):
        raise StandaloneDecoderError("embedding table geometry changed")
    if not embedding_table.dtype.is_floating_point:
        raise StandaloneDecoderError("embedding table must be floating point")
    if not torch.isfinite(embedding_table).all().item():
        raise StandaloneDecoderError("embedding table is non-finite")


def _checked_runtime_embeddings(
    embedding_table: torch.Tensor,
    *,
    hidden_size: int,
    vocab_size: int,
) -> torch.Tensor:
    # Full-table finiteness validation is deliberately performed once by
    # validate_embedding_table at the loaded-resource boundary.  Keep only
    # cheap geometry/dtype checks here so every decoder call does not scan or
    # synchronize a ~1 GiB table.
    if embedding_table.ndim != 2 or tuple(embedding_table.shape) != (vocab_size, hidden_size):
        raise StandaloneDecoderError("runtime embedding table geometry changed")
    if not embedding_table.dtype.is_floating_point:
        raise StandaloneDecoderError("runtime embedding table must be floating point")
    return embedding_table.float()


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(labels).float().mean().detach().cpu())


def _evaluate_decoder(
    model: nn.Module,
    activations: torch.Tensor,
    labels: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    x = activations.to(device=device, dtype=torch.float32)
    y = labels.to(device=device, dtype=torch.long)
    embeddings = embedding_table.to(device=device, dtype=torch.float32)
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            stop = min(start + batch_size, int(x.shape[0]))
            logits = model(x[start:stop], embeddings)
            target = y[start:stop]
            total_loss += float(
                F.cross_entropy(logits, target, reduction="sum").detach().cpu()
            )
            total_correct += int(logits.argmax(dim=-1).eq(target).sum().detach().cpu())
            total_rows += stop - start
    if total_rows == 0:
        raise StandaloneDecoderError("decoder evaluation received no rows")
    return total_loss / total_rows, total_correct / total_rows


def train_token_decoder(
    model: nn.Module,
    activations: torch.Tensor,
    labels: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    config: DecoderTrainingConfig,
    device: torch.device,
    eval_sets: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Fit one tied decoder with full-vocabulary CE and learning curves.

    ``eval_sets`` is intended for public development records.  It is never
    consulted by prediction code and does not affect the optimizer or stopping
    rule.  The complete fixed step schedule is always run.
    """

    config.validate()
    validate_training_tensors(activations, labels, embedding_table)
    for name, pair in (eval_sets or {}).items():
        if len(pair) != 2:
            raise StandaloneDecoderError(f"evaluation set {name} is malformed")
        validate_training_tensors(pair[0], pair[1], embedding_table)
    model = model.to(device)
    x = activations.detach().to(device=device, dtype=torch.float32)
    y = labels.detach().to(device=device, dtype=torch.long)
    embeddings = embedding_table.detach().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    losses: list[float] = []
    gradient_norms: list[float] = []
    curves: list[dict[str, Any]] = []
    model.train()
    for step in range(config.steps):
        indices = torch.randint(
            0,
            x.shape[0],
            (min(config.batch_size, x.shape[0]),),
            generator=generator,
        ).to(device)
        logits = model(x.index_select(0, indices), embeddings)
        loss = F.cross_entropy(logits, y.index_select(0, indices))
        if not torch.isfinite(loss).item():
            raise StandaloneDecoderError("token decoder loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))
        if step == 0 or (step + 1) % config.log_every == 0 or step + 1 == config.steps:
            train_loss, train_accuracy = _evaluate_decoder(
                model, x, y, embeddings, batch_size=config.batch_size
            )
            point: dict[str, Any] = {
                "step": step + 1,
                "train_loss": train_loss,
                "train_token_accuracy": train_accuracy,
            }
            for name, (eval_activation, eval_labels) in (eval_sets or {}).items():
                eval_loss, eval_accuracy = _evaluate_decoder(
                    model,
                    eval_activation,
                    eval_labels,
                    embeddings,
                    batch_size=config.batch_size,
                )
                point[f"{name}_loss"] = eval_loss
                point[f"{name}_token_accuracy"] = eval_accuracy
            curves.append(point)
    model.eval()
    return model, {
        "config": asdict(config),
        "examples": int(x.shape[0]),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "gradient_norm_max": max(gradient_norms),
        "learning_curve": curves,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_state_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
    }


def train_angular_control(
    activations: torch.Tensor,
    target_embeddings: torch.Tensor,
    *,
    config: InverseTrainingConfig,
    device: torch.device,
) -> tuple[ResidualAffineInverse, dict[str, Any]]:
    """Train the existing angular-loss inverse through its canonical API."""

    return train_inverse(activations, target_embeddings, config=config, device=device)



def train_angular_control_with_curve(
    activations: torch.Tensor,
    labels: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    config: InverseTrainingConfig,
    device: torch.device,
    eval_sets: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    log_every: int = 25,
) -> tuple[ResidualAffineInverse, dict[str, Any]]:
    """Train the existing angular inverse and record token learning curves.

    The optimizer update is identical to :func:`train_inverse`: an
    identity-plus-affine residual map, normalized cosine loss, AdamW, sampled
    minibatches, and gradient clipping.  Labels are used only for public-data
    target embeddings and diagnostics; no curve value affects the schedule.
    """

    if log_every <= 0 or config.steps <= 0 or config.batch_size <= 0:
        raise StandaloneDecoderError("angular curve training settings are invalid")
    validate_training_tensors(activations, labels, embedding_table)
    for name, pair in (eval_sets or {}).items():
        if len(pair) != 2:
            raise StandaloneDecoderError(f"evaluation set {name} is malformed")
        validate_training_tensors(pair[0], pair[1], embedding_table)
    model = ResidualAffineInverse(int(activations.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    x = activations.detach().to(device=device, dtype=torch.float32)
    y = labels.detach().to(device=device, dtype=torch.long)
    embeddings = embedding_table.detach().to(device=device, dtype=torch.float32)
    targets = F.normalize(embeddings.index_select(0, y), dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    losses: list[float] = []
    gradient_norms: list[float] = []
    curves: list[dict[str, Any]] = []

    def evaluate_curve(
        eval_x: torch.Tensor,
        eval_y: torch.Tensor,
    ) -> tuple[float, float]:
        value = eval_x.to(device=device, dtype=torch.float32)
        target = eval_y.to(device=device, dtype=torch.long)
        expected = F.normalize(embeddings.index_select(0, target), dim=-1)
        estimate = model(value)
        cosine_loss = float((1.0 - (estimate * expected).sum(dim=-1)).mean().detach().cpu())
        token_accuracy = _accuracy(estimate @ embeddings.transpose(0, 1), target)
        return cosine_loss, token_accuracy

    model.train()
    for step in range(config.steps):
        indices = torch.randint(
            0,
            x.shape[0],
            (min(config.batch_size, x.shape[0]),),
            generator=generator,
        ).to(device)
        estimate = model(x.index_select(0, indices))
        loss = (1.0 - (estimate * targets.index_select(0, indices)).sum(dim=-1)).mean()
        if not torch.isfinite(loss).item():
            raise StandaloneDecoderError("angular inverse loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == config.steps:
            train_loss, train_accuracy = evaluate_curve(x, y)
            point: dict[str, Any] = {
                "step": step + 1,
                "train_loss": train_loss,
                "train_token_accuracy": train_accuracy,
            }
            for name, (eval_x, eval_y) in (eval_sets or {}).items():
                eval_loss, eval_accuracy = evaluate_curve(eval_x, eval_y)
                point[f"{name}_loss"] = eval_loss
                point[f"{name}_token_accuracy"] = eval_accuracy
            curves.append(point)
    model.eval()
    return model, {
        "config": {
            "steps": config.steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "seed": config.seed,
            "log_every": log_every,
        },
        "examples": int(x.shape[0]),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "gradient_norm_max": max(gradient_norms),
        "learning_curve": curves,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_state_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
    }


def decoder_method_ids() -> tuple[str, ...]:
    return (TiedAffineTokenDecoder.method_id, ResidualMLPTokenDecoder.method_id)


def decoder_from_method(
    method_id: str,
    *,
    hidden_size: int,
    vocab_size: int,
    logit_scale: float = 16.0,
    bottleneck_size: int = 256,
) -> nn.Module:
    if method_id == TiedAffineTokenDecoder.method_id:
        return TiedAffineTokenDecoder(hidden_size, vocab_size, logit_scale=logit_scale)
    if method_id == ResidualMLPTokenDecoder.method_id:
        return ResidualMLPTokenDecoder(
            hidden_size,
            vocab_size,
            bottleneck_size=bottleneck_size,
            logit_scale=logit_scale,
        )
    raise StandaloneDecoderError(f"unknown standalone decoder method: {method_id}")


def save_token_decoder(model: nn.Module, path: Path, *, method_id: str, metadata: Mapping[str, str]) -> None:
    """Save decoder-only state, excluding the fixed public embedding table."""

    if path.exists() or path.is_symlink():
        raise StandaloneDecoderError(f"decoder artifact already exists: {path}")
    if method_id not in decoder_method_ids():
        raise StandaloneDecoderError("cannot save unknown decoder method")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    schema = AFFINE_SCHEMA if method_id == TiedAffineTokenDecoder.method_id else MLP_SCHEMA
    save_file(
        state,
        path,
        metadata={"schema": schema, "method_id": method_id, **{str(k): str(v) for k, v in metadata.items()}},
    )


def load_token_decoder(
    path: Path,
    *,
    method_id: str,
    hidden_size: int,
    vocab_size: int,
    device: torch.device,
    logit_scale: float = 16.0,
    bottleneck_size: int = 256,
) -> nn.Module:
    if path.is_symlink() or not path.is_file():
        raise StandaloneDecoderError("decoder artifact must be a regular file")
    state = load_file(path, device="cpu")
    model = decoder_from_method(
        method_id,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        logit_scale=logit_scale,
        bottleneck_size=bottleneck_size,
    )
    expected = set(model.state_dict())
    if set(state) != expected:
        raise StandaloneDecoderError("decoder state fields changed")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def decoder_parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def decoder_source_hash() -> str:
    """Hash this module's source for prediction-to-code binding."""

    path = Path(inspect.getsourcefile(TiedAffineTokenDecoder) or "")
    if not path.is_file():
        raise StandaloneDecoderError("decoder source path cannot be resolved")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash shape, dtype, and contiguous CPU bytes for artifact binding."""

    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(contiguous.shape), "dtype": str(contiguous.dtype)}, sort_keys=True).encode())
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def prediction_tensor(
    model: nn.Module,
    activations: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Emit one direct token per activation row; no candidates or A2 calls."""

    _check_matrix(activations, name="activations")
    model.eval()
    embeddings = embedding_table.to(device=device, dtype=torch.float32)
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(activations.shape[0]), batch_size):
            stop = min(start + batch_size, int(activations.shape[0]))
            logits = model(
                activations[start:stop].to(device=device, dtype=torch.float32),
                embeddings,
            )
            output.append(logits.argmax(dim=-1).to(device="cpu", dtype=torch.int32))
    if not output:
        raise StandaloneDecoderError("prediction received no activation rows")
    result = torch.cat(output, dim=0).contiguous()
    if result.shape[0] != activations.shape[0]:
        raise StandaloneDecoderError("prediction coverage changed")
    return result



def angular_prediction_tensor(
    model: ResidualAffineInverse,
    activations: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Emit direct argmax tokens for the existing angular inverse."""

    _check_matrix(activations, name="activations")
    if batch_size <= 0:
        raise StandaloneDecoderError("prediction batch size must be positive")
    model.eval()
    embeddings = embedding_table.to(device=device, dtype=torch.float32)
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(activations.shape[0]), batch_size):
            stop = min(start + batch_size, int(activations.shape[0]))
            query = model(activations[start:stop].to(device=device, dtype=torch.float32))
            output.append(
                (query @ embeddings.transpose(0, 1)).argmax(dim=-1).to(
                    device="cpu", dtype=torch.int32
                )
            )
    if not output:
        raise StandaloneDecoderError("prediction received no activation rows")
    result = torch.cat(output, dim=0).contiguous()
    if result.shape[0] != activations.shape[0]:
        raise StandaloneDecoderError("prediction coverage changed")
    return result

def save_predictions(
    predictions: Mapping[str, torch.Tensor],
    path: Path,
    *,
    input_sha256: str,
    embedding_sha256: str,
    method_state_hashes: Mapping[str, str],
    code_sha256: str,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write a complete standalone prediction archive with provenance metadata."""

    if path.exists() or path.is_symlink():
        raise StandaloneDecoderError(f"prediction artifact already exists: {path}")
    if not predictions or set(predictions) != set(method_state_hashes):
        raise StandaloneDecoderError("prediction methods and state hashes disagree")
    tensors: dict[str, torch.Tensor] = {}
    for method_id, value in predictions.items():
        if value.ndim != 2 or value.dtype not in (torch.int32, torch.int64):
            raise StandaloneDecoderError(f"prediction tensor malformed: {method_id}")
        if value.shape[0] == 0 or value.lt(0).any().item():
            raise StandaloneDecoderError(f"prediction tensor invalid: {method_id}")
        tensors[f"prediction.{method_id}"] = value.to(torch.int32).cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        path,
        metadata={
            "schema": PREDICTION_SCHEMA,
            "input_sha256": input_sha256,
            "embedding_sha256": embedding_sha256,
            "code_sha256": code_sha256,
            "method_ids": json.dumps(sorted(predictions)),
            "method_state_hashes": json.dumps(dict(sorted(method_state_hashes.items())), sort_keys=True),
            **{str(k): str(v) for k, v in (metadata or {}).items()},
        },
    )


def load_prediction_metadata(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise StandaloneDecoderError("prediction artifact must be a regular file")
    state = load_file(path, device="cpu")
    predictions = {
        key.removeprefix("prediction."): value
        for key, value in state.items()
        if key.startswith("prediction.")
    }
    if not predictions or len(predictions) != len(state):
        raise StandaloneDecoderError("prediction archive contains unexpected tensors")
    # safetensors does not expose metadata through load_file; the script-level
    # scorer reads metadata with safe_open.  This helper is intentionally only
    # for tensor loading and shape checks.
    return predictions, {}


def complete_prediction_check(
    predictions: Mapping[str, torch.Tensor],
    *,
    expected_methods: Iterable[str],
    expected_shape: tuple[int, int],
    vocab_size: int,
) -> None:
    expected = set(expected_methods)
    if set(predictions) != expected:
        raise StandaloneDecoderError("prediction method coverage is incomplete")
    for method_id, value in predictions.items():
        if tuple(value.shape) != expected_shape:
            raise StandaloneDecoderError(f"prediction geometry changed for {method_id}")
        if value.dtype not in (torch.int32, torch.int64):
            raise StandaloneDecoderError(f"prediction dtype changed for {method_id}")
        if value.lt(0).any().item() or value.ge(vocab_size).any().item():
            raise StandaloneDecoderError(f"prediction token range invalid for {method_id}")

