"""Positionwise public-data inverse used by the TRR-0001 direct baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn


class InverseError(RuntimeError):
    """Raised when inverse state or candidate generation is invalid."""


@dataclass(frozen=True)
class InverseTrainingConfig:
    steps: int = 300
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    seed: int = 1731


class ResidualAffineInverse(nn.Module):
    """An identity-plus-affine map into normalized public embedding space."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.residual = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.shape[-1] != self.hidden_size:
            raise InverseError("activation hidden size changed")
        value = activation.float()
        return F.normalize(value + self.residual(value), dim=-1)


def normalized_embeddings(weight: torch.Tensor) -> torch.Tensor:
    value = F.normalize(weight.detach().float(), dim=-1)
    if not torch.isfinite(value).all().item():
        raise InverseError("public embeddings are non-finite")
    return value


def train_inverse(
    activations: torch.Tensor,
    target_embeddings: torch.Tensor,
    *,
    config: InverseTrainingConfig,
    device: torch.device,
) -> tuple[ResidualAffineInverse, dict[str, Any]]:
    """Train from permitted public-surrogate pairs with a fixed schedule."""

    if activations.ndim != 2 or target_embeddings.shape != activations.shape:
        raise InverseError("inverse training tensors must be matching [tokens, hidden]")
    if activations.shape[0] == 0 or config.steps <= 0 or config.batch_size <= 0:
        raise InverseError("inverse training configuration is empty")
    model = ResidualAffineInverse(int(activations.shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    x = activations.detach().to(device=device, dtype=torch.float32)
    target = F.normalize(
        target_embeddings.detach().to(device=device, dtype=torch.float32), dim=-1
    )
    losses: list[float] = []
    model.train()
    for _ in range(config.steps):
        indices = torch.randint(
            0,
            x.shape[0],
            (min(config.batch_size, x.shape[0]),),
            generator=generator,
        ).to(device)
        prediction = model(x.index_select(0, indices))
        loss = (1.0 - (prediction * target.index_select(0, indices)).sum(dim=-1)).mean()
        if not torch.isfinite(loss).item():
            raise InverseError("inverse training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return model, {
        "steps": config.steps,
        "examples": int(x.shape[0]),
        "batch_size": min(config.batch_size, int(x.shape[0])),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "gradient_clip_norm": config.gradient_clip_norm,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def save_inverse(model: ResidualAffineInverse, path: Path, *, cut_depth: int) -> None:
    if path.exists() or path.is_symlink():
        raise InverseError(f"inverse artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "residual.weight": model.residual.weight.detach().cpu().contiguous(),
            "residual.bias": model.residual.bias.detach().cpu().contiguous(),
        },
        path,
        metadata={
            "schema": "token-reconstruction.residual-affine-inverse.v1",
            "hidden_size": str(model.hidden_size),
            "cut_depth": str(cut_depth),
        },
    )


def load_inverse(path: Path, *, hidden_size: int, device: torch.device) -> ResidualAffineInverse:
    if path.is_symlink() or not path.is_file():
        raise InverseError("inverse artifact must be a regular file")
    state = load_file(path, device="cpu")
    if set(state) != {"residual.weight", "residual.bias"}:
        raise InverseError("inverse state fields changed")
    model = ResidualAffineInverse(hidden_size)
    model.residual.load_state_dict(
        {"weight": state["residual.weight"], "bias": state["residual.bias"]},
        strict=True,
    )
    model.requires_grad_(False)
    return model.to(device).eval()


@torch.inference_mode()
def topk_candidates(
    queries: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    k: int,
    score_batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frozen token proposals and cosine scores on CPU."""

    if queries.ndim != 2 or embedding_table.ndim != 2:
        raise InverseError("candidate tensors must be matrices")
    if queries.shape[1] != embedding_table.shape[1] or not 0 < k <= embedding_table.shape[0]:
        raise InverseError("candidate geometry or budget is invalid")
    ids: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    embeddings = embedding_table
    for start in range(0, queries.shape[0], score_batch_size):
        query = F.normalize(queries[start : start + score_batch_size].float(), dim=-1)
        scores = query @ embeddings.transpose(0, 1)
        score, token = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)
        ids.append(token.cpu())
        values.append(score.float().cpu())
    return torch.cat(ids, dim=0), torch.cat(values, dim=0)
