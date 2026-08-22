from __future__ import annotations

import torch
from torch import nn

from token_reconstruction.inverse import (
    InverseTrainingConfig,
    ResidualAffineInverse,
    normalized_embeddings,
    topk_candidates,
    train_inverse,
)
from token_reconstruction.target_update import (
    LoRALinear,
    TargetLoRAConfig,
    set_target_lora_enabled,
)


def test_topk_candidates_and_inverse_training_are_well_formed() -> None:
    torch.manual_seed(7)
    embeddings = normalized_embeddings(torch.eye(5))
    queries = torch.stack((embeddings[3], embeddings[1]))
    ids, scores = topk_candidates(queries, embeddings, k=3, score_batch_size=1)
    assert ids[:, 0].tolist() == [3, 1]
    assert scores.shape == (2, 3)

    x = torch.randn(24, 5)
    target = x.clone()
    model, evidence = train_inverse(
        x,
        target,
        config=InverseTrainingConfig(steps=3, batch_size=8, seed=9),
        device=torch.device("cpu"),
    )
    assert isinstance(model, ResidualAffineInverse)
    assert evidence["steps"] == 3
    assert evidence["trainable_parameters"] == 30


def test_lora_enable_flag_separates_public_and_target_outputs() -> None:
    base = nn.Linear(4, 3, bias=False)
    base.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(11)
    lora = LoRALinear(base, rank=2, alpha=4.0, generator=generator)
    value = torch.randn(2, 4)

    set_target_lora_enabled([lora], False)
    public = lora(value)
    assert torch.equal(public, base(value))

    with torch.no_grad():
        lora.B.fill_(0.2)
    set_target_lora_enabled([lora], True)
    target = lora(value)
    assert not torch.equal(target, public)
    assert TargetLoRAConfig().scale == 2.0
