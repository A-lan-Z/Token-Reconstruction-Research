"""Row-wise CGLS with a latched relative normal-residual convergence gate.

This is a new numerical rule, not an execution-equivalent change to old runs.
"""
import torch
def least_squares(matvec,rmatvec,b,steps,damping=0.0001):
    tiny=torch.finfo(b.dtype).tiny;relative_tolerance=32*torch.finfo(b.dtype).eps
    value=torch.zeros_like(b);residual=b.clone();gradient=rmatvec(residual)
    initial_energy=gradient.square().sum(-1)
    ridge=damping*initial_energy/b.square().sum(-1).clamp_min(tiny)
    threshold=initial_energy*relative_tolerance**2
    energy=initial_energy;active=energy>threshold
    search=torch.where(active[:,None],gradient,torch.zeros_like(gradient))
    frozen_iteration=torch.full((len(b),),steps,device=b.device,dtype=torch.int64)
    frozen_iteration=torch.where(active,frozen_iteration,torch.zeros_like(frozen_iteration))
    for iteration in range(steps):
        response=matvec(search)
        denominator=response.square().sum(-1)+ridge*search.square().sum(-1)
        safe_denominator=torch.where(active,denominator,torch.ones_like(denominator)).clamp_min(tiny)
        alpha=torch.where(active,energy/safe_denominator,torch.zeros_like(energy))
        value=value+alpha[:,None]*search
        residual=residual-alpha[:,None]*response
        gradient=rmatvec(residual)-ridge[:,None]*value
        next_energy=gradient.square().sum(-1)
        next_active=active&(next_energy>threshold)
        newly_frozen=active&(~next_active)
        frozen_iteration=torch.where(newly_frozen,iteration+1,frozen_iteration)
        safe_energy=torch.where(active,energy,torch.ones_like(energy)).clamp_min(tiny)
        beta=torch.where(next_active,next_energy/safe_energy,torch.zeros_like(energy))
        search=torch.where(next_active[:,None],gradient+beta[:,None]*search,torch.zeros_like(search))
        energy=next_energy;active=next_active
    return value,{"recurrence_residual_norm":residual.norm(dim=-1),"ridge":ridge,
      "jvp_calls":steps,"vjp_calls":steps+1,"normal_relative_tolerance":relative_tolerance,
      "converged":~active,"frozen_iteration":frozen_iteration}
