"""Original full-vocabulary step rule with tiled scalar evaluations."""
import torch
from tiled_evaluate import TiledEvaluator
ITERATIONS=16
BUDGET_MARGIN=1e-4
NEWTON_UNDERSHOOT=1e-3

@torch.no_grad()
def path_update(logits, probability_gradient, kl_scale, iterations=16, evaluator_factory=TiledEvaluator):
    logp=torch.log_softmax(logits,dim=-1)
    p=torch.softmax(logits,dim=-1)
    base=torch.logsumexp(logp,dim=-1)
    low_g=probability_gradient.amin(-1,keepdim=True)
    span=(probability_gradient.amax(-1,keepdim=True)-low_g)
    tiny=torch.finfo(logits.dtype).tiny
    d=(probability_gradient-low_g)/span.clamp_min(tiny)
    centered=d-(p*d).sum(-1,keepdim=True)
    correction=(p*centered).sum(-1)
    min_g,index=centered.min(-1)
    min_logp=logp.gather(1,index[:,None])[:,0]
    slope=correction-min_g
    active=(span[:,0]>tiny)&(slope>tiny)
    target=(torch.full_like(slope,kl_scale*(1-BUDGET_MARGIN)) if isinstance(kl_scale,(float,int)) else kl_scale*(1-BUDGET_MARGIN))
    active=active&(target>tiny)
    upper=(target-min_logp+base)/slope.clamp_min(tiny)
    upper=(upper*(1+8*torch.finfo(logits.dtype).eps)).clamp(max=torch.finfo(logits.dtype).max/32)
    upper=torch.where(active,upper,torch.zeros_like(upper))
    initial_upper=upper.clone()
    variance=(p*centered.square()).sum(-1)-correction.square()
    local=torch.sqrt(2*target/variance.clamp_min(tiny))
    point=torch.minimum(local,upper*.5)
    lower=torch.zeros_like(upper)
    lower_kl=torch.zeros_like(upper)
    evaluate=evaluator_factory(logp,centered,base,correction)
    upper_kl,_=evaluate(upper)
    for _ in range(iterations):
        kl,derivative=evaluate(point)
        feasible=(kl<=target)&torch.isfinite(kl)
        lower=torch.where(feasible,point,lower)
        lower_kl=torch.where(feasible,kl,lower_kl)
        upper=torch.where(feasible,upper,point)
        newton=(point-(kl-target)/derivative.clamp_min(tiny))*(1-NEWTON_UNDERSHOOT)
        safe=torch.isfinite(newton)&(newton>lower)&(newton<upper)
        point=torch.where(safe,newton,(lower+upper)*.5)
    shifted=logp-lower[:,None]*centered
    shifted=shifted-shifted.amax(-1,keepdim=True)
    shifted=torch.where(active[:,None],shifted,logits-logits.amax(-1,keepdim=True))
    return shifted,{"normalized_step":lower,"gradient_span":span[:,0],
        "weighted_normalized_variance":variance,"active":active,"target":target,
        "formula_kl":lower_kl,"initial_upper":initial_upper,"initial_upper_kl":upper_kl,
        "final_upper":upper,"centering_correction":correction,
        "budget_used":lower_kl/(kl_scale if isinstance(kl_scale,(float,int)) else kl_scale.clamp_min(tiny))}

@torch.no_grad()
def update(logits, probability_gradient, observed_error, factor, iterations, evaluator_factory=TiledEvaluator):
    budget=(factor*observed_error).clamp(0.,1.)
    new,info=path_update(logits,probability_gradient,budget,iterations,evaluator_factory)
    info["requested_budget"]=budget
    return new,info
