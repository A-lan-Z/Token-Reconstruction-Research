"""Fixed-budget local Krylov direction with a CUDA-capturable small solve."""
import torch
def direction(matvec,b,steps,ridge=1e-6):
    beta=b.norm(dim=-1);eps=torch.finfo(b.dtype).eps
    basis=[b/beta.clamp_min(eps)[:,None]]
    h=torch.zeros((len(b),steps+1,steps),device=b.device,dtype=b.dtype)
    for k in range(steps):
        w=matvec(basis[k])
        for _ in range(2):
            for j in range(k+1):
                coefficient=(basis[j]*w).sum(-1)
                h[:,j,k]+=coefficient;w=w-coefficient[:,None]*basis[j]
        norm=w.norm(dim=-1);h[:,k+1,k]=norm
        basis.append(w/norm.clamp_min(eps)[:,None])
    rhs=torch.zeros((len(b),steps+1,1),device=b.device,dtype=b.dtype);rhs[:,0,0]=beta
    gram=h.transpose(-1,-2)@h
    scale=gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1.)
    regularized=gram+ridge*scale[:,None,None]*torch.eye(steps,device=b.device,dtype=b.dtype)
    coefficient,info=torch.linalg.solve_ex(regularized,h.transpose(-1,-2)@rhs,check_errors=False)
    coefficient=coefficient[...,0]
    value=(torch.stack(basis[:-1],dim=-1)*coefficient[:,None]).sum(-1)
    projected_residual=(h@coefficient[...,None]-rhs).norm(dim=-2)[...,0]
    return value,projected_residual,info
