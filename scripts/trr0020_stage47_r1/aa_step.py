"""Bounded one-history Anderson mixing in full-vocabulary logit coordinates."""
import torch
@torch.no_grad()
def mix(current,proposal,probability,old_residual,old_proposal,valid,metric,clip,ridge=1e-4):
    residual=proposal-current if metric=="logit" else torch.softmax(proposal,dim=-1)-probability
    delta=residual-old_residual
    rr=residual.square().sum();pp=old_residual.square().sum();dd=delta.square().sum()
    denominator=dd+ridge*(rr+pp)
    coefficient=(residual*delta).sum()/denominator.clamp_min(torch.finfo(current.dtype).tiny)
    active=valid & torch.isfinite(coefficient) & (denominator>torch.finfo(current.dtype).tiny)
    coefficient=torch.where(active,coefficient.clamp(-clip,clip),torch.zeros_like(coefficient))
    mixed=proposal-coefficient*(proposal-old_proposal)
    mixed=mixed-mixed.amax(-1,keepdim=True)
    output=torch.where(active,mixed,proposal)
    return output,residual,{"coefficient":coefficient,"residual_squared_norm":rr,"denominator":denominator,"active":active}
