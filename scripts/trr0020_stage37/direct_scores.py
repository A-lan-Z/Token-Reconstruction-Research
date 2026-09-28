"""Direct scores for every cached response; no A2 candidate calls."""
import torch

def norms_for_shift(table,norms,shift):
    dot=table@shift
    energy=shift.square().sum()
    square=norms+2*dot+energy
    bound=64*torch.finfo(table.dtype).eps*(norms+2*norms.sqrt()*energy.sqrt()+energy)
    unstable=square<=bound
    # Stabilize arithmetic only; every row remains in both final score vectors.
    indices=unstable.nonzero().flatten()
    if len(indices):
        square=square.clone()
        square[indices]=(table.index_select(0,indices)+shift).square().sum(-1)
    return square.clamp_min(0),indices

def scores(table,corrected_norms,shift,target,unstable_indices):
    dot=table@target+torch.dot(shift,target)
    if len(unstable_indices):
        dot=dot.clone()
        dot[unstable_indices]=((table.index_select(0,unstable_indices)+shift)*target).sum(-1)
    cosine=dot/(corrected_norms.sqrt().clamp_min(1e-12)*target.norm().clamp_min(1e-12))
    negative_distance=2*dot-corrected_norms-target.square().sum()
    return cosine,negative_distance
