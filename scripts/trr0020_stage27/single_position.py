"""Single-position prefix map with immutable committed past K/V."""
import torch
from torch.nn import functional as F
from linearized_prefix import rms,rms_jvp,rotate,split_heads,merge_heads
from adjoint import rms_vjp,rotate_transpose

def forward(x,layers,past,cos,sin):
    if x.ndim!=2 or len(x)!=1:raise ValueError("one current position required")
    h=x;caches=[];current=[]
    for p,history in zip(layers,past):
        inp,inverse=rms(h,p['n1'],p['eps1'])
        q=rotate(split_heads(F.linear(inp,p['q']),p['heads'],p['head_dim']),cos,sin)
        key=rotate(split_heads(F.linear(inp,p['k']),p['kv_heads'],p['head_dim']),cos,sin)
        value=split_heads(F.linear(inp,p['v']),p['kv_heads'],p['head_dim'])
        k=torch.cat([history[0],key],dim=1).repeat_interleave(p['groups'],dim=0)
        v=torch.cat([history[1],value],dim=1).repeat_interleave(p['groups'],dim=0)
        probability=torch.softmax((q@k.transpose(-1,-2))*p['scale'],dim=-1)
        middle=h+F.linear(merge_heads(probability@v),p['o'])
        normalized,inverse2=rms(middle,p['n2'],p['eps2'])
        gate=F.linear(normalized,p['gate']);up=F.linear(normalized,p['up']);activation=F.silu(gate)
        output=middle+F.linear(activation*up,p['down'])
        caches.append({'h':h,'inverse':inverse,'q':q,'k':k,'v':v,'attention':probability,
          'middle':middle,'inverse2':inverse2,'gate':gate,'up':up,'activation':activation})
        current.append((key,value));h=output
    return h,caches,current

def jvp(tangent,layers,caches,cos,sin):
    dh=tangent
    for p,c in zip(layers,caches):
        dinp=rms_jvp(c['h'],dh,p['n1'],c['inverse'])
        dq=rotate(split_heads(F.linear(dinp,p['q']),p['heads'],p['head_dim']),cos,sin)
        dk=rotate(split_heads(F.linear(dinp,p['k']),p['kv_heads'],p['head_dim']),cos,sin).repeat_interleave(p['groups'],dim=0)
        dv=split_heads(F.linear(dinp,p['v']),p['kv_heads'],p['head_dim']).repeat_interleave(p['groups'],dim=0)
        ds=(dq@c['k'].transpose(-1,-2))*p['scale']
        ds=torch.cat([ds[...,:-1],ds[...,-1:]+(c['q']*dk).sum(-1,keepdim=True)*p['scale']],dim=-1)
        probability=c['attention'];dp=probability*(ds-(probability*ds).sum(-1,keepdim=True))
        dout=dp@c['v']+probability[...,-1:]*dv
        dmid=dh+F.linear(merge_heads(dout),p['o'])
        dn=rms_jvp(c['middle'],dmid,p['n2'],c['inverse2'])
        dg=F.linear(dn,p['gate']);du=F.linear(dn,p['up'])
        sigmoid=torch.sigmoid(c['gate']);prime=sigmoid*(1+c['gate']*(1-sigmoid))
        dh=dmid+F.linear(prime*dg*c['up']+c['activation']*du,p['down'])
    return dh

def vjp(gradient,layers,caches,cos,sin):
    value=gradient
    for p,c in reversed(list(zip(layers,caches))):
        g_product=value@p['down'];sigmoid=torch.sigmoid(c['gate']);prime=sigmoid*(1+c['gate']*(1-sigmoid))
        g_normalized=(g_product*prime*c['up'])@p['gate']+(g_product*c['activation'])@p['up']
        g_middle=value+rms_vjp(c['middle'],g_normalized,p['n2'],c['inverse2'])
        g_attention=split_heads(g_middle@p['o'],p['heads'],p['head_dim'])
        probability=c['attention'];g_probability=g_attention@c['v'].transpose(-1,-2)
        g_scores=probability*(g_probability-(probability*g_probability).sum(-1,keepdim=True))
        g_query=(g_scores@c['k'])*p['scale']
        g_key=g_scores[...,-1:]*c['q']*p['scale'];g_value=probability[...,-1:]*g_attention
        g_key=g_key.reshape(p['kv_heads'],p['groups'],1,p['head_dim']).sum(1)
        g_value=g_value.reshape(p['kv_heads'],p['groups'],1,p['head_dim']).sum(1)
        g_input=(merge_heads(rotate_transpose(g_query,cos,sin))@p['q']
          +merge_heads(rotate_transpose(g_key,cos,sin))@p['k']+merge_heads(g_value)@p['v'])
        value=g_middle+rms_vjp(c['h'],g_input,p['n1'],c['inverse'])
    return value

def commit(past,current):
    return [(torch.cat([old[0],new[0]],dim=1),torch.cat([old[1],new[1]],dim=1)) for old,new in zip(past,current)]
