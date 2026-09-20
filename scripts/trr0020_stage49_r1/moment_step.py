"""Full-vocabulary KL steps from bounded moment-generating functions."""
from pathlib import Path
import sys,torch
sys.path.append(str(Path(__file__).resolve().parents[1]/"trr0020_stage44"))
from pointwise import TorchPointwise,TritonPointwise
MODES=("bennett","two_point")
ROOT_ITERATIONS=48
BUDGET_MARGIN=1e-4
RATE_MARGIN=1e-6

def expm1_minus(x):
    y=x.clamp(-.01,.01)
    polynomial=y.square()*(.5+y*(1/6+y*(1/24+y*(1/120+y/720))))
    return torch.where(x.abs()<.01,polynomial,torch.expm1(x)-x)

def bound_value(s,c,mode):
    if mode=="bennett":return torch.log1p(c*expm1_minus(s))
    alpha=c/(1+c);beta=1/(1+c)
    return torch.log1p(alpha*expm1_minus(s)+beta*expm1_minus(-c*s))

def solve_torch(variance,positive_bound,target,active,mode):
    variance=variance.double();b=positive_bound.double();target=target.double()
    b=torch.where(active,b,torch.ones_like(b))
    variance=torch.where(active,variance,torch.ones_like(variance))
    target=torch.where(active,target,torch.zeros_like(target))
    c=variance/b.square()
    if mode=="bennett":
        q=torch.expm1(target)/c
        upper=torch.minimum(torch.sqrt(2*q),torch.log1p(q)+1)
    elif mode=="two_point":upper=target+torch.log1p(1/c)
    else:raise ValueError("unknown bound")
    lower=torch.zeros_like(upper)
    for _ in range(ROOT_ITERATIONS):
        middle=(lower+upper)*.5
        feasible=bound_value(middle,c,mode)<=target
        lower=torch.where(feasible,middle,lower);upper=torch.where(feasible,upper,middle)
    rate=(lower/b)*(1-RATE_MARGIN)
    rate=torch.where(active,rate,torch.zeros_like(rate))
    bound=bound_value(rate*b,c,mode)
    return rate,bound

@torch.no_grad()
def update(logits,gradient,error,mode,probability=None,gpu_root=True,pw=None):
    if mode not in MODES:raise ValueError("unknown mode")
    if pw is None:pw=TritonPointwise if logits.is_cuda else TorchPointwise
    p=torch.softmax(logits,-1) if probability is None else probability
    tiny=torch.finfo(logits.dtype).tiny
    mass=p.sum(-1)
    low=gradient.amin(-1,keepdim=True)
    span=gradient.amax(-1,keepdim=True)-low
    d,pd=pw.normalized(gradient,low,span.clamp_min(tiny),p)
    mean=pd.sum(-1)/mass
    centered,pc,pcc=pw.centered(d,mean[:,None],p)
    correction=pc.sum(-1).double()/mass.double()
    variance=(pcc.sum(-1).double()/mass.double()-correction.square()).clamp_min(0.)
    positive=correction-centered.amin(-1).double()
    budget=(2*error).clamp(0.,1.)
    target=budget.double()*(1-BUDGET_MARGIN)
    active=(span[:,0]>tiny)&(variance>tiny)&(positive>tiny)&(target>tiny)
    if logits.is_cuda and gpu_root:
        from scalar_kernel import solve
        rate,bound=solve(variance,positive,target,active,mode)
    else:rate,bound=solve_torch(variance,positive,target,active,mode)
    rate=rate.clamp(max=torch.finfo(logits.dtype).max/32).to(logits.dtype)
    raw=pw.tilt(logits,rate,centered)
    shifted=pw.finish(raw,raw.amax(-1,keepdim=True),logits,logits.amax(-1,keepdim=True),active)
    return shifted,{"normalized_step":rate,"requested_budget":budget,"bound_kl":bound,
      "weighted_variance":variance,"positive_bound":positive,"active":active,
      "centering_correction":correction,"gradient_span":span[:,0],"target":target}
