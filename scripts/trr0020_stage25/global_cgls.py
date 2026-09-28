"""Coupled least squares with one global inner product per sequence.

Row-wise coefficients are invalid for a Jacobian coupling token positions.
"""
import torch
def least_squares(matvec,rmatvec,b,steps,damping=0.0001):
    tiny=torch.finfo(b.dtype).tiny
    value=torch.zeros_like(b);residual=b.clone();gradient=rmatvec(residual)
    initial_energy=gradient.square().sum()
    ridge=damping*initial_energy/b.square().sum().clamp_min(tiny)
    search=gradient.clone();energy=initial_energy
    for _ in range(steps):
        response=matvec(search)
        denominator=response.square().sum()+ridge*search.square().sum()
        alpha=energy/denominator.clamp_min(tiny)
        value=value+alpha*search
        residual=residual-alpha*response
        gradient=rmatvec(residual)-ridge*value
        next_energy=gradient.square().sum()
        beta=next_energy/energy.clamp_min(tiny)
        search=gradient+beta*search
        energy=next_energy
    return value,{"recurrence_residual_norm":residual.norm(dim=-1),
      "global_recurrence_residual_norm":residual.norm(),"ridge":ridge,
      "jvp_calls":steps,"vjp_calls":steps+1}
