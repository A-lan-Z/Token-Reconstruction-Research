"""Configurable execution of a contiguous public Llama prefix."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import torch
from torch import nn


class PublicPrefixError(RuntimeError):
    """Raised when the public-prefix executor cannot preserve cache semantics."""


@dataclass
class PublicPrefixCache:
    """Cache state whose committed length is checked on every transition."""

    backend: Any
    length: int = 0


class ContiguousPublicPrefix(nn.Module):
    """Embedding plus decoder layers `[0, cut_depth)` from a public Llama model."""

    def __init__(self, full_model: nn.Module, cut_depth: int) -> None:
        super().__init__()
        inner = getattr(full_model, "model", None)
        if inner is None or not hasattr(inner, "layers"):
            raise PublicPrefixError("model does not expose a Llama-style decoder")
        layer_count = len(inner.layers)
        if not isinstance(cut_depth, int) or isinstance(cut_depth, bool):
            raise PublicPrefixError("cut_depth must be an integer")
        if cut_depth < 0 or cut_depth >= layer_count:
            raise PublicPrefixError(
                f"cut_depth must be between 0 and {layer_count - 1} so a downstream layer remains"
            )

        self.embed_tokens = inner.embed_tokens
        self.layers = nn.ModuleList(list(inner.layers[:cut_depth]))
        self.rotary_emb = inner.rotary_emb
        self.config = full_model.config
        self.cut_depth = cut_depth
        self.cache_keyword: str | None = None

        if self.layers:
            parameters = inspect.signature(self.layers[0].forward).parameters
            if "past_key_values" in parameters:
                self.cache_keyword = "past_key_values"
            elif "past_key_value" in parameters:
                self.cache_keyword = "past_key_value"
            else:
                raise PublicPrefixError("decoder layer exposes no supported cache argument")
            for layer in self.layers[1:]:
                if self.cache_keyword not in inspect.signature(layer.forward).parameters:
                    raise PublicPrefixError("public prefix layers disagree on their cache API")

    def new_cache(self) -> PublicPrefixCache:
        if self.cut_depth == 0:
            return PublicPrefixCache(backend=None)
        from transformers.cache_utils import DynamicCache

        try:
            backend = DynamicCache(config=self.config)
        except TypeError:  # Transformers 4.x compatibility.
            backend = DynamicCache()
        return PublicPrefixCache(backend=backend)

    def _cache_layer_lengths(self, cache: PublicPrefixCache) -> tuple[int, ...]:
        if self.cut_depth == 0:
            return ()
        getter = getattr(cache.backend, "get_seq_length", None)
        if not callable(getter):
            raise PublicPrefixError("public cache does not expose get_seq_length")
        lengths: list[int] = []
        for layer_index in range(self.cut_depth):
            try:
                value = getter(layer_index)
            except TypeError:  # Legacy caches expose one global length.
                value = lengths[0] if lengths else getter()
            except IndexError:
                value = 0
            lengths.append(int(value))
        return tuple(lengths)

    def _require_cache_length(
        self, cache: PublicPrefixCache, expected: int, where: str
    ) -> None:
        if cache.length != expected:
            raise PublicPrefixError(
                f"logical cache length failed {where}: {cache.length} != {expected}"
            )
        observed = self._cache_layer_lengths(cache)
        if any(length != expected for length in observed):
            raise PublicPrefixError(
                f"decoder cache lengths failed {where}: {observed} != {expected}"
            )

    @staticmethod
    def _causal_mask(
        hidden: torch.Tensor, *, start_pos: int, total_tokens: int
    ) -> torch.Tensor | None:
        query_tokens = int(hidden.shape[1])
        if query_tokens == 1:
            return None
        minimum = torch.finfo(hidden.dtype).min
        mask = torch.full(
            (query_tokens, total_tokens),
            minimum,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        mask = torch.triu(mask, diagonal=1 + start_pos)
        return mask.view(1, 1, query_tokens, total_tokens).expand(
            hidden.shape[0], 1, query_tokens, total_tokens
        )

    @staticmethod
    def _hidden(output: Any) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output

    @torch.inference_mode()
    def run_cached(
        self,
        input_ids: torch.Tensor,
        cache: PublicPrefixCache,
        start_pos: int,
    ) -> torch.Tensor:
        """Commit one contiguous token block and return its cut activation."""

        if input_ids.ndim != 2 or input_ids.shape[1] <= 0:
            raise PublicPrefixError("input_ids must be non-empty [batch, time]")
        if start_pos < 0:
            raise PublicPrefixError("start_pos must be non-negative")
        self._require_cache_length(cache, start_pos, "before commit")

        hidden = self.embed_tokens(input_ids)
        batch, tokens, _ = hidden.shape
        if self.cut_depth == 0:
            cache.length = start_pos + tokens
            return hidden

        position_ids = torch.arange(
            start_pos, start_pos + tokens, dtype=torch.long, device=hidden.device
        ).view(1, -1).expand(batch, -1)
        cache_position = torch.arange(
            start_pos, start_pos + tokens, dtype=torch.long, device=hidden.device
        )
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = self._causal_mask(
            hidden, start_pos=start_pos, total_tokens=start_pos + tokens
        )
        assert self.cache_keyword is not None
        for layer in self.layers:
            output = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **{self.cache_keyword: cache.backend},
            )
            hidden = self._hidden(output)

        cache.length = start_pos + tokens
        self._require_cache_length(cache, start_pos + tokens, "after commit")
        return hidden

    @torch.inference_mode()
    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a complete unpadded token block without retaining a cache."""

        if input_ids.ndim != 2 or input_ids.shape[1] <= 0:
            raise PublicPrefixError("input_ids must be non-empty [batch, time]")
        hidden = self.embed_tokens(input_ids)
        if self.cut_depth == 0:
            return hidden

        batch, tokens, _ = hidden.shape
        position_ids = torch.arange(
            tokens, dtype=torch.long, device=hidden.device
        ).view(1, -1).expand(batch, -1)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = self._causal_mask(hidden, start_pos=0, total_tokens=tokens)
        for layer in self.layers:
            output = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden = self._hidden(output)
        return hidden
