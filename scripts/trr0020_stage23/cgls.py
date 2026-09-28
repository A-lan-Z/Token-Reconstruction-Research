"""Fixed-budget damped least squares using Jacobian and transpose products."""
import torch
def least_squares(matvec,rmatvec,b,steps,damping=0.):
    tiny=torch.finfo(b.dtype).tiny
    value=torch.zeros_like(b);residual=b.clone();gradient=rmatvec(residual)
    initial_energy=gradient.square().sum(-1)
    scale=initial_energy/b.square().sum(-1).clamp_min(tiny)
    ridge=damping*scale
    search=gradient.clone();energy=initial_energy
    for _ in range(steps):
        response=matvec(search)
        denominator=response.square().sum(-1)+ridge*search.square().sum(-1)
        alpha=energy/denominator.clamp_min(tiny)
        value=value+alpha[:,None]*search
        residual=residual-alpha[:,None]*response
        gradient=rmatvec(residual)-ridge[:,None]*value
        next_energy=gradient.square().sum(-1)
        beta=next_energy/energy.clamp_min(tiny)
        search=gradient+beta[:,None]*search
        energy=next_energy
    return value,{"recurrence_residual_norm":residual.norm(dim=-1),"ridge":ridge,"jvp_calls":steps,"vjp_calls":steps+1}
