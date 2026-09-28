"""Full causal prefix derivatives, including cross-position attention paths."""
import torch
from torch.nn import functional as F
from linearized_prefix import rms_jvp,rotate,split_heads,merge_heads
from adjoint import rms_vjp,rotate_transpose

def full_jvp(tangent,layers,cache,cos,sin):
    dh=tangent
    for p,c in zip(layers,cache):
        dinp=rms_jvp(c['h'],dh,p['n1'],c['inverse'])
        dq=rotate(split_heads(F.linear(dinp,p['q']),p['heads'],p['head_dim']),cos,sin)
        dk=rotate(split_heads(F.linear(dinp,p['k']),p['kv_heads'],p['head_dim']),cos,sin).repeat_interleave(p['groups'],dim=0)
        dv=split_heads(F.linear(dinp,p['v']),p['kv_heads'],p['head_dim']).repeat_interleave(p['groups'],dim=0)
        ds=(dq@c['k'].transpose(-1,-2)+c['q']@dk.transpose(-1,-2))*p['scale']
        attention=c['attention'];da=attention*(ds-(attention*ds).sum(-1,keepdim=True))
        dout=da@c['v']+attention@dv
        dmid=dh+F.linear(merge_heads(dout),p['o'])
        dn=rms_jvp(c['middle'],dmid,p['n2'],c['inverse2'])
        dg=F.linear(dn,p['gate']);du=F.linear(dn,p['up'])
        sigmoid=torch.sigmoid(c['gate']);prime=sigmoid*(1+c['gate']*(1-sigmoid))
        dh=dmid+F.linear(prime*dg*c['up']+c['activation']*du,p['down'])
    return dh

def full_vjp(gradient,layers,cache,cos,sin):
    value=gradient
    for p,c in reversed(list(zip(layers,cache))):
        g_product=value@p["down"]
        sigmoid=torch.sigmoid(c["gate"]);prime=sigmoid*(1+c["gate"]*(1-sigmoid))
        g_normalized=(g_product*prime*c["up"])@p["gate"]+(g_product*c["activation"])@p["up"]
        g_middle=value+rms_vjp(c["middle"],g_normalized,p["n2"],c["inverse2"])
        g_attention=split_heads(g_middle@p["o"],p["heads"],p["head_dim"])
        probability=c["attention"];g_probability=g_attention@c["v"].transpose(-1,-2)
        g_scores=probability*(g_probability-(probability*g_probability).sum(-1,keepdim=True))
        g_query=(g_scores@c["k"])*p["scale"]
        g_key=(g_scores.transpose(-1,-2)@c["q"])*p["scale"]
        g_value=probability.transpose(-1,-2)@g_attention
        length=g_key.shape[1]
        g_key=g_key.reshape(p["kv_heads"],p["groups"],length,p["head_dim"]).sum(1)
        g_value=g_value.reshape(p["kv_heads"],p["groups"],length,p["head_dim"]).sum(1)
        g_input=(merge_heads(rotate_transpose(g_query,cos,sin))@p["q"]
          +merge_heads(rotate_transpose(g_key,cos,sin))@p["k"]+merge_heads(g_value)@p["v"])
        value=g_middle+rms_vjp(c["h"],g_input,p["n1"],c["inverse"])
    return value

def free_positions(value):
    """Orthogonal projection onto inputs other than declared fixed BOS."""
    return torch.cat([torch.zeros_like(value[:1]),value[1:]],dim=0)

def operators(layers,cache,cos,sin):
    return (lambda value:full_jvp(free_positions(value),layers,cache,cos,sin),
            lambda value:free_positions(full_vjp(value,layers,cache,cos,sin)))
