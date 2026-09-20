"""Full-vocabulary BOS-response decision, without proposals or verification."""
import torch
def select(table,lengths,target):
    scores=(table@target)/lengths.clamp_min(1e-12)/target.norm().clamp_min(1e-12)
    return scores.argmax().reshape(1),scores
