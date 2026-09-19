"""FP32/FP64 Llama-style forward and own-position Jacobian-vector products.

Prototype only: the CPU reference qualifies algebra, not the production prefix.
The forward is a new numerical implementation, not native BF16 equivalence.
"""
import torch
from torch.nn import functional as F

def rms(x,weight,eps):
    inverse=torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)
    return x*inverse*weight,inverse

def rms_jvp(x,dx,weight,inverse):
    return (dx-x*(x*dx).mean(-1,keepdim=True)*inverse.square())*inverse*weight

def rotate(x,cos,sin):
    half=x.shape[-1]//2
    return x*cos[None]+torch.cat([-x[...,half:],x[...,:half]],dim=-1)*sin[None]

def split_heads(x,heads,dim):return x.view(x.shape[0],heads,dim).transpose(0,1)
def merge_heads(x):return x.transpose(0,1).contiguous().flatten(1)

def forward(x,layers,cos,sin):
    """Unpadded causal forward, preserving the supplied dtype."""
    h=x;cache=[];length=len(x)
    forbidden=torch.ones((length,length),dtype=torch.bool,device=x.device).triu(1)
    for p in layers:
        inp,inverse=rms(h,p['n1'],p['eps1'])
        q=rotate(split_heads(F.linear(inp,p['q']),p['heads'],p['head_dim']),cos,sin)
        k=rotate(split_heads(F.linear(inp,p['k']),p['kv_heads'],p['head_dim']),cos,sin)
        v=split_heads(F.linear(inp,p['v']),p['kv_heads'],p['head_dim'])
        k=k.repeat_interleave(p['groups'],dim=0);v=v.repeat_interleave(p['groups'],dim=0)
        scores=(q@k.transpose(-1,-2))*p['scale'];scores=scores.masked_fill(forbidden[None],float('-inf'))
        attention=torch.softmax(scores,dim=-1)
        middle=h+F.linear(merge_heads(attention@v),p['o'])
        normalized,inverse2=rms(middle,p['n2'],p['eps2'])
        gate=F.linear(normalized,p['gate']);up=F.linear(normalized,p['up']);activation=F.silu(gate)
        output=middle+F.linear(activation*up,p['down'])
        cache.append({'h':h,'inverse':inverse,'q':q,'k':k,'v':v,'attention':attention,'middle':middle,'inverse2':inverse2,'gate':gate,'up':up,'activation':activation})
        h=output
    return h,cache

def diagonal_jvp(tangent,layers,cache,cos,sin):
    """Exact real-arithmetic diagonal blocks d(output_i)/d(input_i).

Past positions' K/V derivatives are excluded; current query, key and value
paths remain. This is not the full sequence Jacobian-vector product.
"""
    dh=tangent
    for p,c in zip(layers,cache):
        dinp=rms_jvp(c['h'],dh,p['n1'],c['inverse'])
        dq=rotate(split_heads(F.linear(dinp,p['q']),p['heads'],p['head_dim']),cos,sin)
        dk=rotate(split_heads(F.linear(dinp,p['k']),p['kv_heads'],p['head_dim']),cos,sin).repeat_interleave(p['groups'],dim=0)
        dv=split_heads(F.linear(dinp,p['v']),p['kv_heads'],p['head_dim']).repeat_interleave(p['groups'],dim=0)
        ds=(dq@c['k'].transpose(-1,-2)+torch.diag_embed((c['q']*dk).sum(-1)))*p['scale']
        attention=c['attention'];da=attention*(ds-(attention*ds).sum(-1,keepdim=True))
        dout=da@c['v']+attention.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)*dv
        dmid=dh+F.linear(merge_heads(dout),p['o'])
        dn=rms_jvp(c['middle'],dmid,p['n2'],c['inverse2'])
        dg=F.linear(dn,p['gate']);du=F.linear(dn,p['up'])
        sigmoid=torch.sigmoid(c['gate']);silu_prime=sigmoid*(1+c['gate']*(1-sigmoid))
        dh=dmid+F.linear(silu_prime*dg*c['up']+c['activation']*du,p['down'])
    return dh

def krylov_least_squares(matvec,b,steps=8,ridge=1e-6):
    """Batched reorthogonalized Arnoldi with a regularized small least-squares solve.

Each row of b is an independent block. Fixed iteration budget, no token
ranking or vocabulary selection. One final matvec reports residual size.
"""
    beta=b.norm(dim=-1);eps=torch.finfo(b.dtype).eps
    basis=[b/beta.clamp_min(eps)[:,None]]
    h=torch.zeros((len(b),steps+1,steps),device=b.device,dtype=b.dtype)
    for k in range(steps):
        w=matvec(basis[k])
        for repeat in range(2):
            for j in range(k+1):
                coefficient=(basis[j]*w).sum(-1)
                h[:,j,k]+=coefficient;w=w-coefficient[:,None]*basis[j]
        norm=w.norm(dim=-1);h[:,k+1,k]=norm
        basis.append(w/norm.clamp_min(eps)[:,None])
    rhs=torch.zeros((len(b),steps+1,1),device=b.device,dtype=b.dtype);rhs[:,0,0]=beta
    gram=h.transpose(-1,-2)@h
    scale=gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1.)
    regularized=gram+ridge*scale[:,None,None]*torch.eye(steps,device=b.device,dtype=b.dtype)
    coefficients=torch.linalg.solve(regularized,h.transpose(-1,-2)@rhs)[...,0]
    direction=(torch.stack(basis[:-1],dim=-1)*coefficients[:,None]).sum(-1)
    residual=(matvec(direction)-b).norm(dim=-1)
    return direction,{'residual_norm':residual,'rhs_norm':beta,'matvec_calls':steps+1}

def public_parameters(prefix):
    """Prepare frozen FP32 copies; intended only for the declared Llama prefix."""
    layers=[]
    for layer in prefix.layers:
        attention=layer.self_attn;mlp=layer.mlp
        modules=[attention.q_proj,attention.k_proj,attention.v_proj,attention.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj]
        if any(module.bias is not None for module in modules):raise ValueError('bias architecture not qualified')
        heads=attention.q_proj.out_features//attention.head_dim;kv_heads=attention.k_proj.out_features//attention.head_dim
        p={key:module.weight.detach().float().contiguous() for key,module in zip(['q','k','v','o','gate','up','down'],modules)}
        p.update(n1=layer.input_layernorm.weight.detach().float(),n2=layer.post_attention_layernorm.weight.detach().float(),eps1=layer.input_layernorm.variance_epsilon,eps2=layer.post_attention_layernorm.variance_epsilon,heads=heads,kv_heads=kv_heads,head_dim=attention.head_dim,groups=heads//kv_heads,scale=attention.scaling)
        layers.append(p)
    return layers
