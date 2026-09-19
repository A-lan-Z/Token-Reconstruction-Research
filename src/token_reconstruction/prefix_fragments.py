"""No-fit token proposals from the current prefix and tokenizer decomposition."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .prefix_weight_metric import PrefixWeightMetric

def suffix_ids(tokenizer, token_id):
    """First token of each proper character suffix, in deterministic order."""
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if "\ufffd" in text:
        return ()
    result = []
    for offset in range(1, len(text)):
        ids = tokenizer.encode(text[offset:], add_special_tokens=False)
        if ids and ids[0] != token_id and ids[0] not in result:
            result.append(ids[0])
    return tuple(result)

def build_suffix_cache(tokenizer, vocabulary_size, chunk=4096):
    """Cache tokenizer metadata only; no observations or model responses."""
    result = []
    for start in range(0, vocabulary_size, chunk):
        token_ids = list(range(start, min(start + chunk, vocabulary_size)))
        texts = tokenizer.batch_decode([[v] for v in token_ids], skip_special_tokens=False)
        suffixes, spans = [], []
        for text in texts:
            lo = len(suffixes)
            if "\ufffd" not in text:
                suffixes.extend(text[j:] for j in range(1, len(text)))
            spans.append((lo, len(suffixes)))
        encoded = tokenizer(suffixes, add_special_tokens=False, return_attention_mask=False)["input_ids"] if suffixes else []
        for token_id, (lo, hi) in zip(token_ids, spans):
            unique = []
            for ids in encoded[lo:hi]:
                if ids and ids[0] != token_id and ids[0] not in unique:
                    unique.append(ids[0])
            result.append(tuple(unique))
    return result

def expand_candidates(base, suffix_cache, budget=512):
    """Preserve the base; add suffixes round-robin; pad with first candidate."""
    if not base or budget < len(base):
        raise ValueError("budget must accommodate the nonempty base")
    result = list(dict.fromkeys(int(v) for v in base))
    seen = set(result)
    fragments = [suffix_cache[int(v)] for v in base]
    for depth in range(max((len(f) for f in fragments), default=0)):
        for fragment in fragments:
            if depth < len(fragment) and fragment[depth] not in seen:
                result.append(fragment[depth])
                seen.add(fragment[depth])
                if len(result) >= budget:
                    return result[:budget]
    return result + [result[0]] * (budget - len(result))

@torch.no_grad()
def intrinsic_table(prefix, chunk=256):
    """Exact declared MLP-only forward map; attention is omitted by definition."""
    raw = torch.empty_like(prefix.embed_tokens.weight)
    for start in range(0, len(raw), chunk):
        hidden = prefix.embed_tokens.weight[start:start + chunk].unsqueeze(1)
        for layer in prefix.layers:
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        raw[start:start + len(hidden)] = hidden[:, 0]
    return raw

class PrefixFragmentProposer:
    def __init__(self, prefix, suffix_cache):
        self.prefix = prefix
        self.suffix_cache = suffix_cache
        self.metric = PrefixWeightMetric(prefix)
        self.dictionary = None

    @torch.no_grad()
    def build(self, native_chunk=256):
        stats = self.metric.build()
        raw = intrinsic_table(self.prefix, native_chunk)
        self.dictionary = torch.empty_like(self.metric.table)
        # Keep this projection geometry identical to the development definition.
        for start in range(0, len(raw), 256):
            self.dictionary[start:start + 256] = F.normalize(
                raw[start:start + 256].float() @ self.metric.transform, dim=-1)
        self.metric._check()
        return {**stats, "intrinsic_cache_bytes": self.dictionary.numel() * self.dictionary.element_size(),
                "native_chunk": native_chunk, "projection_chunk": 256, "fit_steps": 0, "fit_examples": 0}

    @torch.no_grad()
    def base(self, observations):
        self.metric._check()
        if self.dictionary is None:
            raise RuntimeError("build prefix-derived tables first")
        first = self.metric.propose(observations, 64)
        query = F.normalize(observations.to(self.dictionary.device).float() @ self.metric.transform, dim=-1)
        second = (query @ self.dictionary.T).topk(64, dim=-1).indices
        return torch.cat([first, second], dim=-1)

    @torch.no_grad()
    def propose(self, observations, fragments=True, budget=512):
        base = self.base(observations)
        if not fragments:
            return base
        expanded = [expand_candidates(row, self.suffix_cache, budget=budget) for row in base.cpu().tolist()]
        return torch.tensor(expanded, device=base.device, dtype=torch.long)
