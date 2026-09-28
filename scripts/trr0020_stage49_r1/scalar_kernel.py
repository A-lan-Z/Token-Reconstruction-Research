"""Only per-position FP64 root arithmetic; all vocabulary reductions stay native."""
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from moment_step import ROOT_ITERATIONS,RATE_MARGIN
@triton.jit
def _expm1_minus(x):
    y=tl.minimum(tl.maximum(x,-.01),.01)
    polynomial=y*y*(.5+y*(1/6+y*(1/24+y*(1/120+y/720))))
    return tl.where(tl.abs(x)<.01,polynomial,lib.expm1(x)-x)
@triton.jit
def _value(s,c,MODE:tl.constexpr):
    if MODE==0:return lib.log1p(c*_expm1_minus(s))
    alpha=c/(1+c);beta=1/(1+c)
    return lib.log1p(alpha*_expm1_minus(s)+beta*_expm1_minus(-c*s))
@triton.jit
def _solve(V,B,T,A,R,K,N:tl.constexpr,MODE:tl.constexpr,IT:tl.constexpr,SHRINK:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    active=tl.load(A+i,mask,False)
    v=tl.load(V+i,mask,1.);b=tl.load(B+i,mask,1.);target=tl.load(T+i,mask,0.)
    v=tl.where(active,v,1.);b=tl.where(active,b,1.);target=tl.where(active,target,0.)
    c=v/(b*b)
    if MODE==0:
        q=lib.expm1(target)/c
        upper=tl.minimum(lib.sqrt(2*q),lib.log1p(q)+1)
    else:upper=target+lib.log1p(1/c)
    lower=tl.full((BLOCK,),0.,tl.float64)
    for _ in range(IT):
        middle=(lower+upper)*.5
        feasible=_value(middle,c,MODE)<=target
        lower=tl.where(feasible,middle,lower);upper=tl.where(feasible,upper,middle)
    rate=tl.where(active,(lower/b)*SHRINK,0.)
    bound=_value(rate*b,c,MODE)
    tl.store(R+i,rate,mask);tl.store(K+i,bound,mask)
def solve(variance,bound,target,active,mode):
    rate=torch.empty_like(variance);value=torch.empty_like(variance)
    _solve[(triton.cdiv(variance.numel(),128),)](variance,bound,target,active,rate,value,variance.numel(),
      0 if mode=="bennett" else 1,ROOT_ITERATIONS,1-RATE_MARGIN,128,num_warps=4,enable_fp_fusion=False)
    return rate,value
