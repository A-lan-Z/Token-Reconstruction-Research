"""Fixed prefix-derived input metrics; no fitted predictor or token shortlist."""
import torch
from cheap_update import direction as original_direction
METRICS=("identity","metric","inverse_metric")
RULES=("polyak0.25","cg1")

@torch.no_grad()
def build(transform):
    if not torch.isfinite(transform).all():raise ValueError("nonfinite transform")
    gram=transform@transform.T
    gram=(gram+gram.T)*.5
    scale=gram.diagonal().mean()
    if not torch.isfinite(scale) or scale<=0:raise ValueError("degenerate transform")
    matrix=gram+1e-3*scale*torch.eye(len(gram),device=gram.device,dtype=gram.dtype)
    factor,info=torch.linalg.cholesky_ex(matrix)
    if int(info)!=0:raise RuntimeError("metric is not positive definite")
    inverse=torch.cholesky_inverse(factor)
    inverse=(inverse+inverse.T)*.5
    result={"identity":None,"metric":matrix/matrix.diagonal().median(),
      "inverse_metric":inverse/inverse.diagonal().median()}
    return result,{"regularizer":float(1e-3*scale),"matrix":matrix,"inverse":inverse}

def direction(rule,rhs,matvec,rmatvec,preconditioner=None):
    if rule not in RULES:raise ValueError("unregistered rule")
    if preconditioner is None:
        value,work=original_direction(rule,rhs,matvec,rmatvec)
        return value,{**work,"metric_products":0}
    tiny=torch.finfo(rhs.dtype).tiny
    gradient=rmatvec(rhs);search=gradient@preconditioner.T
    energy=(gradient*search).sum(-1)
    if not torch.isfinite(energy).all():raise RuntimeError("nonfinite metric direction")
    active=energy>tiny
    safe_energy=torch.where(active,energy,torch.ones_like(energy))
    if rule=="polyak0.25":
        alpha=torch.where(active,.25*rhs.square().sum(-1)/safe_energy,torch.zeros_like(energy))
        return alpha[:,None]*search,{"jvp_calls":0,"vjp_calls":1,"metric_products":1}
    ridge=.0001*gradient.square().sum(-1)/rhs.square().sum(-1).clamp_min(tiny)
    search=torch.where(active[:,None],search,torch.zeros_like(search))
    response=matvec(search)
    denominator=response.square().sum(-1)+ridge*search.square().sum(-1)
    safe=torch.where(active,denominator,torch.ones_like(denominator)).clamp_min(tiny)
    alpha=torch.where(active,energy/safe,torch.zeros_like(energy))
    return alpha[:,None]*search,{"jvp_calls":1,"vjp_calls":1,"metric_products":1}
