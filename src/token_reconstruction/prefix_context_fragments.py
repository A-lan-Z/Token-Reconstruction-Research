"""Prioritize tokenizer continuations compatible with already reconstructed text."""
from __future__ import annotations
from .prefix_fragments import expand_candidates


def build_context_cache(tokenizer, vocabulary_size, chunk=4096):
    """One tokenizer-only pass builds the original suffix list and exact prefixes."""
    suffix_cache, context_cache = [], []
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
        for token_id, text, (lo, hi) in zip(token_ids, texts, spans):
            unique, pairs = [], []
            for offset, ids in enumerate(encoded[lo:hi], 1):
                if ids and ids[0] != token_id:
                    pairs.append((text[:offset], int(ids[0])))
                    if ids[0] not in unique:
                        unique.append(int(ids[0]))
            suffix_cache.append(tuple(unique))
            context_cache.append(tuple(pairs))
    return suffix_cache, context_cache


def contextual_candidates(base, suffix_cache, context_cache, reconstructed_text, budget=256):
    """Keep every base token; promote exact context matches; retain old fallback."""
    if not base or budget < len(base):
        raise ValueError("budget must accommodate the nonempty base")
    result = list(dict.fromkeys(int(v) for v in base))
    seen = set(result)
    compatible = []
    for rank, token_id in enumerate(base):
        for removed, candidate in context_cache[int(token_id)]:
            if reconstructed_text.endswith(removed):
                compatible.append((-len(removed), rank, candidate))
    compatible.sort()
    for _, _, candidate in compatible:
        if candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
            if len(result) == budget:
                return result
    for candidate in expand_candidates(base, suffix_cache, budget):
        if candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
            if len(result) == budget:
                return result
    return result + [result[0]] * (budget - len(result))