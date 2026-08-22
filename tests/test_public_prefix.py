from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from token_reconstruction.public_prefix import ContiguousPublicPrefix, PublicPrefixError


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 5)
        self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.model.rotary_emb = nn.Identity()
        self.config = SimpleNamespace()


def test_zero_layer_cut_is_exact_embedding_and_tracks_length() -> None:
    model = FakeModel()
    prefix = ContiguousPublicPrefix(model, cut_depth=0)
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    assert torch.equal(prefix.forward_full(tokens), model.model.embed_tokens(tokens))
    cache = prefix.new_cache()
    assert torch.equal(prefix.run_cached(tokens[:, :2], cache, 0), model.model.embed_tokens(tokens[:, :2]))
    assert cache.length == 2
    prefix.run_cached(tokens[:, 2:], cache, 2)
    assert cache.length == 3


def test_cache_start_position_is_fail_closed() -> None:
    prefix = ContiguousPublicPrefix(FakeModel(), cut_depth=0)
    cache = prefix.new_cache()
    with pytest.raises(PublicPrefixError, match="logical cache length"):
        prefix.run_cached(torch.tensor([[1]]), cache, start_pos=1)
