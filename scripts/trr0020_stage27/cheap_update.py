"""Current-token directions requiring one transpose, optionally one forward product."""
import torch
RULES=("polyak0.25","polyak0.5","polyak1.0","cg1")
def direction(rule,rhs,matvec,rmatvec):
    if rule not in RULES:raise ValueError("unknown current-token update")
    tiny=torch.finfo(rhs.dtype).tiny
    gradient=rmatvec(rhs);energy=gradient.square().sum(-1)
    if rule=="cg1":
        active=energy>energy*(32*torch.finfo(rhs.dtype).eps)**2
        search=torch.where(active[:,None],gradient,torch.zeros_like(gradient))
        ridge=.0001*energy/rhs.square().sum(-1).clamp_min(tiny)
        response=matvec(search)
        denominator=response.square().sum(-1)+ridge*search.square().sum(-1)
        safe=torch.where(active,denominator,torch.ones_like(denominator)).clamp_min(tiny)
        alpha=torch.where(active,energy/safe,torch.zeros_like(energy))
        return alpha[:,None]*search,{"jvp_calls":1,"vjp_calls":1}
    gamma=float(rule.removeprefix("polyak"))
    active=energy>tiny
    safe=torch.where(active,energy,torch.ones_like(energy))
    alpha=torch.where(active,gamma*rhs.square().sum(-1)/safe,torch.zeros_like(energy))
    return alpha[:,None]*gradient,{"jvp_calls":0,"vjp_calls":1}
