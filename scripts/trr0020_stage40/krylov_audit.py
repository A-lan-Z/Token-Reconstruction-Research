"""Instrumented Arnoldi with unchanged FP32 normal and stable reduced solves."""
import torch

def arnoldi(matvec,b,steps):
    eps=torch.finfo(b.dtype).eps;beta=b.norm();basis=[b/beta.clamp_min(eps)]
    h=torch.zeros((steps+1,steps),device=b.device,dtype=b.dtype)
    for k in range(steps):
        w=matvec(basis[k])
        for _ in range(2):
            for j in range(k+1):
                coefficient=(basis[j]*w).sum();h[j,k]+=coefficient;w=w-coefficient*basis[j]
        norm=w.norm();h[k+1,k]=norm;basis.append(w/norm.clamp_min(eps))
    return torch.stack(basis,-1),h,beta

def reduced(h,beta,mode):
    steps=h.shape[1]
    rhs=torch.zeros((steps+1,1),device=h.device,dtype=h.dtype);rhs[0,0]=beta
    gram=h.T@h;scale=gram.diagonal().mean().clamp_min(1.)
    if mode=="normal_ridge":
        coefficient=torch.linalg.solve(gram+1e-6*scale*torch.eye(steps,device=h.device,dtype=h.dtype),h.T@rhs)[:,0]
    else:
        hh=h.cpu().double();bb=rhs.cpu().double()
        if mode=="svd_ridge":
            hh=torch.cat([hh,(1e-6*scale.cpu().double()).sqrt()*torch.eye(steps,dtype=torch.float64)])
            bb=torch.cat([bb,torch.zeros((steps,1),dtype=torch.float64)])
        elif mode!="svd_unregularized":raise ValueError(mode)
        coefficient=torch.linalg.lstsq(hh,bb,driver="gelsd",rcond=1e-12).solution[:,0].to(device=h.device)
    return coefficient

def direction(basis,h,beta,steps,mode):
    small=h[:steps+1,:steps].contiguous()
    coefficient=reduced(small,beta,mode)
    answer=((basis[...,:steps]*coefficient).sum(-1) if mode=="normal_ridge" else (basis[...,:steps].double()*coefficient).sum(-1).to(basis.dtype))
    rhs=torch.zeros(steps+1,device=h.device,dtype=h.dtype);rhs[0]=beta
    return answer,coefficient,(small@coefficient-rhs if mode=="normal_ridge" else small.double()@coefficient-rhs.double())
