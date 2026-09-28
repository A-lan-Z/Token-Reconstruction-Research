"""Bounded one-history Anderson mixing in full-vocabulary logit coordinates."""
import torch
@torch.no_grad()
def mix(current,proposal,probability,old_residual,old_proposal,valid,metric,clip,ridge=1e-4):
    residual=proposal-current if metric=="logit" else torch.softmax(proposal,dim=-1)-probability
    tiny=torch.finfo(current.dtype).tiny
    scale=torch.maximum(residual.abs().amax(),old_residual.abs().amax()).clamp_min(tiny) if metric=="logit" else torch.ones((),dtype=current.dtype,device=current.device)
    r=residual/scale if metric=="logit" else residual
    old=old_residual/scale if metric=="logit" else old_residual
    delta=r-old
    rr=r.square().sum();pp=old.square().sum();dd=delta.square().sum()
    denominator=dd+ridge*(rr+pp)
    coefficient=(r*delta).sum()/denominator.clamp_min(torch.finfo(current.dtype).tiny)
    active=valid & torch.isfinite(coefficient) & (denominator>torch.finfo(current.dtype).tiny)
    coefficient=torch.where(active,coefficient.clamp(-clip,clip),torch.zeros_like(coefficient))
    mixed=proposal-coefficient*(proposal-old_proposal)
    mixed=mixed-mixed.amax(-1,keepdim=True)
    mixed_finite=torch.isfinite(mixed).all()
    active=active & mixed_finite
    coefficient=torch.where(active,coefficient,torch.zeros_like(coefficient))
    output=torch.where(active,mixed,proposal)
    return output,residual,{"coefficient":coefficient,"residual_squared_norm":rr,"denominator":denominator,"active":active,"scale":scale,"mixed_finite":mixed_finite}
