"""Analytic transpose of the own-position causal Jacobian."""
import torch
from linearized_prefix import split_heads,merge_heads
def rms_vjp(x,gradient,weight,inverse):
    scaled=gradient*weight*inverse
    return scaled-x*(x*scaled).mean(-1,keepdim=True)*inverse.square()
def rotate_transpose(x,cos,sin):
    half=x.shape[-1]//2
    scaled=x*sin[None]
    return x*cos[None]-torch.cat([-scaled[...,half:],scaled[...,:half]],dim=-1)
def diagonal_vjp(gradient,layers,cache,cos,sin):
    value=gradient
    for p,c in reversed(list(zip(layers,cache))):
        g_product=value@p["down"]
        sigmoid=torch.sigmoid(c["gate"]);prime=sigmoid*(1+c["gate"]*(1-sigmoid))
        g_normalized=(g_product*prime*c["up"])@p["gate"]+(g_product*c["activation"])@p["up"]
        g_middle=value+rms_vjp(c["middle"],g_normalized,p["n2"],c["inverse2"])
        g_attention=split_heads(g_middle@p["o"],p["heads"],p["head_dim"])
        g_probability=g_attention@c["v"].transpose(-1,-2)
        probability=c["attention"]
        g_scores=probability*(g_probability-(probability*g_probability).sum(-1,keepdim=True))
        g_query=(g_scores@c["k"])*p["scale"]
        g_key=g_scores.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)*c["q"]*p["scale"]
        g_value=probability.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)*g_attention
        length=g_key.shape[1]
        g_key=g_key.reshape(p["kv_heads"],p["groups"],length,p["head_dim"]).sum(1)
        g_value=g_value.reshape(p["kv_heads"],p["groups"],length,p["head_dim"]).sum(1)
        g_input=(merge_heads(rotate_transpose(g_query,cos,sin))@p["q"]
          +merge_heads(rotate_transpose(g_key,cos,sin))@p["k"]
          +merge_heads(g_value)@p["v"])
        value=g_middle+rms_vjp(c["h"],g_input,p["n1"],c["inverse"])
    return value
