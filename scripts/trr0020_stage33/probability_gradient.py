"""Gradient over the entire vocabulary for one token with a fixed causal history."""
import torch
@torch.no_grad()
def gradient(output,target,embedding,current_input_vjp,eps=1e-12):
    if output.ndim!=2 or len(output)!=1 or target.shape!=output.shape:
        raise ValueError("one current position required")
    raw_norm=output.norm(dim=-1,keepdim=True)
    norm=raw_norm.clamp_min(eps)
    unit=output/norm
    target_unit=target/target.norm(dim=-1,keepdim=True).clamp_min(eps)
    alignment=(unit*target_unit).sum(-1,keepdim=True)
    cotangent=(-target_unit+unit*alignment*(raw_norm>eps))/norm
    input_gradient=current_input_vjp(cotangent)
    vocabulary_gradient=input_gradient@embedding.T
    return vocabulary_gradient,{"loss":1-alignment[:,0],"output_cotangent":cotangent,
        "input_gradient":input_gradient}
