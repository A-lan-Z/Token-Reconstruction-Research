"""Current-input right preconditioners for the qualified coupled Jacobian."""
import torch

KINDS=("identity","input_norm","position_probe4","coordinate_probe4")

def make_scale(kind,x,rmatvec,probes):
    if kind not in KINDS:raise ValueError("unknown scaling rule")
    tiny=torch.finfo(x.dtype).tiny
    if kind=="identity":
        scale=torch.ones_like(x);calls=0
    elif kind=="input_norm":
        scale=x.norm(dim=-1,keepdim=True).clamp_min(tiny).expand_as(x);calls=0
    else:
        if len(probes)!=4:raise ValueError("exactly four fixed probes required")
        energy=torch.zeros_like(x)
        for probe in probes:energy.add_(rmatvec(probe).square()/4)
        position=energy.mean(-1,keepdim=True)
        floor=position[1:].median().clamp_min(tiny)*1e-6
        diagonal=position.expand_as(x) if kind=="position_probe4" else .5*energy+.5*position
        scale=diagonal.clamp_min(floor.clamp_min(tiny)).rsqrt();calls=4
    scale=scale/scale[1:].median().clamp_min(tiny)
    scale=torch.cat([torch.zeros_like(scale[:1]),scale[1:]],dim=0)
    return scale,{"extra_vjp_calls":calls,"probe_count":calls}

def scaled_operators(matvec,rmatvec,scale):
    return (lambda value:matvec(scale*value),lambda value:scale*rmatvec(value))
