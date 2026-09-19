"""Mirror step whose size shrinks with the current observed cosine mismatch."""
import torch
@torch.no_grad()
def update(logits,probability_gradient,kl_scale,observed_error,max_span=4.):
    probability=torch.softmax(logits,dim=-1)
    centered=probability_gradient-(probability*probability_gradient).sum(-1,keepdim=True)
    variance=(probability*centered.square()).sum(-1)
    span=centered.amax(-1)-centered.amin(-1)
    tiny=torch.finfo(logits.dtype).tiny
    rate=torch.sqrt(2*kl_scale/variance.clamp_min(tiny))
    rate=torch.minimum(rate,max_span/span.clamp_min(tiny))
    active=span>tiny
    rate=torch.where(active,rate,torch.zeros_like(rate))
    factor=(observed_error/.05).clamp(0,1).sqrt()
    rate=rate*factor
    shifted=logits-rate[:,None]*centered
    shifted=shifted-shifted.amax(-1,keepdim=True)
    return shifted,{"rate":rate,"weighted_variance":variance,"gradient_span":span,
      "update_span":rate*span,"active":active,"error_factor":factor}
