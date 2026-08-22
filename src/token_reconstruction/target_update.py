"""Evaluator-only low-rank target-prefix update primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from torch import nn


class TargetUpdateError(RuntimeError):
    """Raised when the evaluator-only update cannot be installed safely."""


@dataclass(frozen=True)
class TargetLoRAConfig:
    layers: tuple[int, ...] = (0, 1, 2, 3)
    modules: tuple[str, ...] = ("q_proj", "v_proj")
    rank: int = 4
    alpha: float = 8.0
    seed: int = 1730

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Frozen base projection plus an evaluator-controlled low-rank delta."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if rank <= 0 or base.in_features <= 0 or base.out_features <= 0:
            raise TargetUpdateError("invalid LoRA geometry")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        initial_a = torch.empty(rank, base.in_features, dtype=torch.float32)
        nn.init.normal_(initial_a, mean=0.0, std=0.01, generator=generator)
        self.A = nn.Parameter(initial_a.to(base.weight.device))
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        self.enabled = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        if not self.enabled:
            return base
        hidden = F.linear(value.float(), self.A)
        delta = F.linear(hidden, self.B).mul(self.scale)
        return base + delta.to(base.dtype)


def install_target_lora(
    model: nn.Module, config: TargetLoRAConfig
) -> dict[str, LoRALinear]:
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise TargetUpdateError("target model is not Llama-style")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    installed: dict[str, LoRALinear] = {}
    for layer_index in config.layers:
        try:
            attention = inner.layers[layer_index].self_attn
        except (AttributeError, IndexError) as exc:
            raise TargetUpdateError(f"target layer {layer_index} is unavailable") from exc
        for module_name in config.modules:
            base = getattr(attention, module_name, None)
            if not isinstance(base, nn.Linear):
                raise TargetUpdateError(
                    f"target module {layer_index}.{module_name} is not linear"
                )
            wrapper = LoRALinear(
                base,
                rank=config.rank,
                alpha=config.alpha,
                generator=generator,
            ).to(base.weight.device)
            setattr(attention, module_name, wrapper)
            installed[f"layers.{layer_index}.self_attn.{module_name}"] = wrapper
    return installed


def set_target_lora_enabled(installed: Iterable[LoRALinear], enabled: bool) -> None:
    for module in installed:
        module.enabled = enabled


def target_lora_parameters(installed: Iterable[LoRALinear]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    for module in installed:
        result.extend((module.A, module.B))
    return result


def save_target_lora(installed: dict[str, LoRALinear], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise TargetUpdateError(f"target update already exists: {path}")
    tensors: dict[str, torch.Tensor] = {}
    for name, module in installed.items():
        tensors[f"{name}.A"] = module.A.detach().cpu().contiguous()
        tensors[f"{name}.B"] = module.B.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        path,
        metadata={
            "schema": "token-reconstruction.evaluator-target-lora.v1",
            "access": "evaluator-only",
        },
    )


def load_target_lora(installed: dict[str, LoRALinear], path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise TargetUpdateError("target update must be a regular file")
    state = load_file(path, device="cpu")
    expected = {
        f"{name}.{suffix}" for name in installed for suffix in ("A", "B")
    }
    if set(state) != expected:
        raise TargetUpdateError("target update fields changed")
    with torch.no_grad():
        for name, module in installed.items():
            module.A.copy_(state[f"{name}.A"].to(module.A.device))
            module.B.copy_(state[f"{name}.B"].to(module.B.device))
