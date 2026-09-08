"""Deterministic truth-free ranking and post-freeze record-level metrics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

BUDGETS = (1, 8, 16, 32, 64, 256)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def verify(b):
    actual=binding(b['path'])
    if any(actual[k]!=b[k] for k in ('path','bytes','sha256')):
        raise ValueError(f"binding changed: {b['path']}")
    return Path(b['path'])


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def rank_scores(scores, max_k=256):
    if scores.ndim != 2 or not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError('scores must be finite [positions,vocabulary] floats')
    if not 1 <= max_k <= scores.shape[-1]:
        raise ValueError('invalid budget')
    # Stable sort sees token IDs in ascending order, including ties at the cutoff.
    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :max_k].contiguous()
    return ids, scores.gather(-1, ids).contiguous()


def rank_truth(candidates, labels):
    if candidates.ndim != 3 or labels.shape != candidates.shape[:2]:
        raise ValueError('candidate/label shape mismatch')
    if candidates.dtype != torch.int64 or labels.dtype != torch.int64:
        raise ValueError('IDs must be int64')
    if torch.any(candidates.sort(-1).values.diff(dim=-1) == 0):
        raise ValueError('duplicate candidate')
    matches = candidates.eq(labels.unsqueeze(-1))
    return torch.where(matches.any(-1), matches.long().argmax(-1) + 1, candidates.shape[-1] + 1).numpy()


def bootstrap_mean(values, seed=9013, draws=10000):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('finite nonempty record values required')
    rng = np.random.default_rng(seed)
    means = a[rng.integers(len(a), size=(draws, len(a)))].mean(1)
    return {'estimate': float(a.mean()), 'paired_source_bootstrap_95': np.quantile(means, [.025, .975]).tolist(), 'records': len(a)}


def metrics(ranks, k):
    r = np.asarray(ranks)
    if r.ndim != 2 or not r.size:
        raise ValueError('nonempty record-position ranks required')
    hit = r <= k
    wrong = r > 1
    rescued = hit & wrong
    # Resample records jointly for the conditional ratio. Keep undefined draws explicit.
    rng = np.random.default_rng(9013)
    ix = rng.integers(len(r), size=(10000, len(r)))
    denom = wrong.sum(1)[ix].sum(1)
    num = rescued.sum(1)[ix].sum(1)
    valid = denom > 0
    conditional = None if not wrong.any() else float(rescued.sum() / wrong.sum())
    return {
        'k': k, 'records': len(r), 'scored_tokens': int(r.size),
        'included_tokens': int(hit.sum()), 'omitted_tokens': int((~hit).sum()),
        'token_recall': bootstrap_mean(hit.mean(1)),
        'complete_clips': int(hit.all(1).sum()), 'clip_coverage': bootstrap_mean(hit.all(1)),
        'clip_coverage_interpretation': 'Perfect fixed-shortlist selector upper bound, not achieved A2 recovery',
        'top1_wrong_tokens': int(wrong.sum()), 'rescued_top1_wrong_tokens': int(rescued.sum()),
        'recall_among_top1_wrong': conditional,
        'conditional_bootstrap_95': None if not valid.all() else np.quantile(num / denom, [.025, .975]).tolist(),
        'conditional_undefined_bootstrap_draws': int((~valid).sum()),
        'per_record_omissions': (~hit).sum(1).tolist(),
    }


def paired(candidate, reference):
    a, b = np.asarray(candidate, dtype=bool), np.asarray(reference, dtype=bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('paired source geometry mismatch')
    from scipy.stats import beta
    harmed = (b & ~a).any(1)
    n, x = len(harmed), int(harmed.sum())
    upper = 1.0 if x == n else float(beta.ppf(.95, x + 1, n - x))
    token = bootstrap_mean(a.mean(1) - b.mean(1))
    clip = bootstrap_mean(a.all(1).astype(float) - b.all(1).astype(float))
    return {'token_delta': token, 'clip_delta': clip,
            'lost_inclusions': int((b & ~a).sum()), 'gained_inclusions': int((a & ~b).sum()),
            'records_with_any_lost_inclusion': x,
            'record_loss_probability_one_sided_95_upper': upper,
            'empirical_candidate_loss_limits_met': token['estimate'] >= -.001 - 1e-12 and clip['estimate'] >= -.02 - 1e-12,
            'equivalence_established': False,
            'uncertainty_note': 'Percentile bootstrap can degenerate at zero losses; it does not establish equivalence. Exact record-loss upper bound retained.'}
