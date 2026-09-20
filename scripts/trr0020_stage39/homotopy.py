"""Residual-strength homotopy and full coupled sequence derivatives."""
import torch
from torch.nn import functional as F
from linearized_prefix import rms,rms_jvp,rotate,split_heads,merge_heads

def forward(x,layers,cos,sin,strength):
    h=x;cache=[];length=len(x)
    forbidden=torch.ones((length,length),dtype=torch.bool,device=x.device).triu(1)
    for p in layers:
        normalized,inverse=rms(h,p["n1"],p["eps1"])
        q=rotate(split_heads(F.linear(normalized,p["q"]),p["heads"],p["head_dim"]),cos,sin)
        k=rotate(split_heads(F.linear(normalized,p["k"]),p["kv_heads"],p["head_dim"]),cos,sin).repeat_interleave(p["groups"],dim=0)
        v=split_heads(F.linear(normalized,p["v"]),p["kv_heads"],p["head_dim"]).repeat_interleave(p["groups"],dim=0)
        attention=torch.softmax(((q@k.transpose(-1,-2))*p["scale"]).masked_fill(forbidden[None],float("-inf")),dim=-1)
        attention_write=F.linear(merge_heads(attention@v),p["o"])
        middle=h+strength*attention_write
        value,inverse2=rms(middle,p["n2"],p["eps2"])
        gate=F.linear(value,p["gate"]);up=F.linear(value,p["up"]);activation=F.silu(gate)
        mlp_write=F.linear(activation*up,p["down"])
        cache.append({"h":h,"inverse":inverse,"q":q,"k":k,"v":v,"attention":attention,
          "attention_write":attention_write,"middle":middle,"inverse2":inverse2,"gate":gate,"up":up,"activation":activation,"mlp_write":mlp_write})
        h=middle+strength*mlp_write
    return h,cache

def jvp(tangent,layers,cache,cos,sin,strength,strength_tangent=0.):
    dh=tangent
    for p,c in zip(layers,cache):
        dn=rms_jvp(c["h"],dh,p["n1"],c["inverse"])
        dq=rotate(split_heads(F.linear(dn,p["q"]),p["heads"],p["head_dim"]),cos,sin)
        dk=rotate(split_heads(F.linear(dn,p["k"]),p["kv_heads"],p["head_dim"]),cos,sin).repeat_interleave(p["groups"],dim=0)
        dv=split_heads(F.linear(dn,p["v"]),p["kv_heads"],p["head_dim"]).repeat_interleave(p["groups"],dim=0)
        ds=(dq@c["k"].transpose(-1,-2)+c["q"]@dk.transpose(-1,-2))*p["scale"]
        a=c["attention"];da=a*(ds-(a*ds).sum(-1,keepdim=True))
        attention_delta=F.linear(merge_heads(da@c["v"]+a@dv),p["o"])
        dmid=dh+strength*attention_delta+strength_tangent*c["attention_write"]
        dn2=rms_jvp(c["middle"],dmid,p["n2"],c["inverse2"])
        dg=F.linear(dn2,p["gate"]);du=F.linear(dn2,p["up"])
        sigmoid=torch.sigmoid(c["gate"]);prime=sigmoid*(1+c["gate"]*(1-sigmoid))
        mlp_delta=F.linear(prime*dg*c["up"]+c["activation"]*du,p["down"])
        dh=dmid+strength*mlp_delta+strength_tangent*c["mlp_write"]
    return dh

def global_gmres(matvec,b,steps,ridge=1e-6):
    """One coupled linear system, with global inner products across positions."""
    eps=torch.finfo(b.dtype).eps;beta=b.norm();basis=[b/beta.clamp_min(eps)]
    h=torch.zeros((steps+1,steps),device=b.device,dtype=b.dtype)
    for k in range(steps):
        w=matvec(basis[k])
        for _ in range(2):
            for j in range(k+1):
                coefficient=(basis[j]*w).sum();h[j,k]+=coefficient;w=w-coefficient*basis[j]
        norm=w.norm();h[k+1,k]=norm;basis.append(w/norm.clamp_min(eps))
    rhs=torch.zeros((steps+1,1),device=b.device,dtype=b.dtype);rhs[0,0]=beta
    gram=h.T@h;scale=gram.diagonal().mean().clamp_min(1.)
    coefficients=torch.linalg.solve(gram+ridge*scale*torch.eye(steps,device=b.device,dtype=b.dtype),h.T@rhs)[:,0]
    answer=(torch.stack(basis[:-1],-1)*coefficients).sum(-1)
    return answer,{"rhs_norm":beta,"residual_norm":(matvec(answer)-b).norm(),"matvec_calls":steps+1}
