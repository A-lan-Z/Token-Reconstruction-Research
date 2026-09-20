"""Second-moment-preserving full-vocabulary embedding interpolation."""
import torch
EPS=1e-12
@torch.no_grad()
def mixture(probability,embedding,norm_squared,eps=EPS):
    mean=probability@embedding
    energy=probability@norm_squared[:,None]
    raw_squared=mean.square().sum(-1,keepdim=True)
    scale=torch.sqrt(energy.clamp_min(eps*eps)/raw_squared.clamp_min(eps*eps))
    return mean*scale,{"mean":mean,"energy":energy,"raw_squared":raw_squared,"scale":scale}
@torch.no_grad()
def gradient(output,target,embedding,norm_squared,state,current_input_vjp,eps=EPS):
    raw_norm=output.norm(dim=-1,keepdim=True);norm=raw_norm.clamp_min(eps)
    unit=output/norm;target_unit=target/target.norm(dim=-1,keepdim=True).clamp_min(eps)
    alignment=(unit*target_unit).sum(-1,keepdim=True)
    cotangent=(-target_unit+unit*alignment*(raw_norm>eps))/norm
    input_gradient=current_input_vjp(cotangent)
    mean,energy,raw_squared,scale=(state[k] for k in ["mean","energy","raw_squared","scale"])
    dot=(input_gradient*mean).sum(-1,keepdim=True)
    mean_gradient=scale*(input_gradient-mean*(dot/raw_squared.clamp_min(eps*eps))*(raw_squared>eps*eps))
    energy_gradient=.5*dot*scale/energy.clamp_min(eps*eps)*(energy>eps*eps)
    probability_gradient=mean_gradient@embedding.T+energy_gradient*norm_squared[None]
    return probability_gradient,{"loss":1-alignment[:,0],"input_gradient":input_gradient,
      "mean_gradient":mean_gradient,"energy_gradient":energy_gradient}
