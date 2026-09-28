"""Invert the own-position Jacobian as an ordered product of layer factors."""
import torch
from krylov_graph import direction
from linearized_prefix import diagonal_jvp
def solve_factors(matvecs,b,steps,ridge=1e-6):
    value=b;diagnostics=[];info=torch.zeros(len(b),device=b.device,dtype=torch.int32)
    for index in reversed(range(len(matvecs))):
        rhs=value
        value,residual,status=direction(matvecs[index],rhs,steps,ridge)
        info=torch.maximum(info,status.abs())
        diagnostics.append(residual/rhs.norm(dim=-1).clamp_min(torch.finfo(b.dtype).eps))
    return value,torch.stack(diagnostics),info
def factorized_direction(layers,cache,cos,sin,b,steps,ridge=1e-6):
    products=[lambda value,p=p,c=c:diagonal_jvp(value,[p],[c],cos,sin) for p,c in zip(layers,cache)]
    return solve_factors(products,b,steps,ridge)
