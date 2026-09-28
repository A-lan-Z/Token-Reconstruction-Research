"""Elementwise kernels with explicit FP32 boundaries; reductions stay native."""
import torch
import triton
import triton.language as tl

@triton.jit
def _normalized(G,LOW,DEN,P,D,PD,N:tl.constexpr,V:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N;row=i//V
    g=tl.load(G+i,mask,0.);low=tl.load(LOW+row,mask,0.);den=tl.load(DEN+row,mask,1.)
    p=tl.load(P+i,mask,0.)
    d=tl.div_rn(g-low,den)
    tl.store(D+i,d,mask);tl.store(PD+i,p*d,mask)

@triton.jit
def _centered(D,MEAN,P,C,PC,PCC,N:tl.constexpr,V:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N;row=i//V
    d=tl.load(D+i,mask,0.);mean=tl.load(MEAN+row,mask,0.);p=tl.load(P+i,mask,0.)
    c=d-mean;square=c*c
    tl.store(C+i,c,mask);tl.store(PC+i,p*c,mask);tl.store(PCC+i,p*square,mask)

@triton.jit
def _tilted(LP,T,C,OUT,N:tl.constexpr,V:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N;row=i//V
    lp=tl.load(LP+i,mask,0.);t=tl.load(T+row,mask,0.);c=tl.load(C+i,mask,0.)
    product=t*c
    tl.store(OUT+i,lp-product,mask)

@triton.jit
def _finish(RAW,RM,L,LM,A,OUT,N:tl.constexpr,V:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N;row=i//V
    raw=tl.load(RAW+i,mask,0.);rm=tl.load(RM+row,mask,0.)
    l=tl.load(L+i,mask,0.);lm=tl.load(LM+row,mask,0.);active=tl.load(A+row,mask,False)
    left=raw-rm;right=l-lm
    tl.store(OUT+i,tl.where(active,left,right),mask)

@triton.jit
def _advance(PT,KL,DER,TARGET,LO,LKL,UP,NEWLO,NEWLKL,NEWUP,NEWPT,
             N:tl.constexpr,TINY:tl.constexpr,SHRINK:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
    pt=tl.load(PT+i,mask,0.);kl=tl.load(KL+i,mask,0.);der=tl.load(DER+i,mask,1.)
    target=tl.load(TARGET+i,mask,0.);lo=tl.load(LO+i,mask,0.);lkl=tl.load(LKL+i,mask,0.);up=tl.load(UP+i,mask,0.)
    finite=(kl==kl)&(tl.abs(kl)!=float("inf"));feasible=(kl<=target)&finite
    lo=tl.where(feasible,pt,lo);lkl=tl.where(feasible,kl,lkl);up=tl.where(feasible,up,pt)
    den=tl.maximum(der,TINY)
    quotient=tl.div_rn(kl-target,den)
    newton=(pt-quotient)*SHRINK
    safe=(newton==newton)&(tl.abs(newton)!=float("inf"))&(newton>lo)&(newton<up)
    midpoint=(lo+up)*.5;point=tl.where(safe,newton,midpoint)
    tl.store(NEWLO+i,lo,mask);tl.store(NEWLKL+i,lkl,mask);tl.store(NEWUP+i,up,mask);tl.store(NEWPT+i,point,mask)

class TorchPointwise:
    @staticmethod
    def normalized(g,low,den,p):
        d=(g-low)/den
        return d,p*d
    @staticmethod
    def centered(d,mean,p):
        c=d-mean
        return c,p*c,p*c.square()
    @staticmethod
    def tilt(lp,t,c):return lp-t[:,None]*c
    @staticmethod
    def finish(raw,rm,logits,lm,active):return torch.where(active[:,None],raw-rm,logits-lm)
    @staticmethod
    def advance(point,kl,derivative,target,lower,lower_kl,upper,tiny,shrink):
        feasible=(kl<=target)&torch.isfinite(kl)
        lower=torch.where(feasible,point,lower);lower_kl=torch.where(feasible,kl,lower_kl);upper=torch.where(feasible,upper,point)
        newton=(point-(kl-target)/derivative.clamp_min(tiny))*shrink
        safe=torch.isfinite(newton)&(newton>lower)&(newton<upper)
        point=torch.where(safe,newton,(lower+upper)*.5)
        return lower,lower_kl,upper,point

class TritonPointwise:
    @staticmethod
    def check(a):
        if a.dtype!=torch.float32 or not a.is_cuda or not a.is_contiguous():raise ValueError("contiguous FP32CUDA required")
    @staticmethod
    def normalized(g,low,den,p):
        TritonPointwise.check(g);d=torch.empty_like(g);pd=torch.empty_like(g)
        _normalized[(triton.cdiv(g.numel(),1024),)](g,low,den,p,d,pd,g.numel(),g.shape[1],1024,num_warps=4,enable_fp_fusion=False)
        return d,pd
    @staticmethod
    def centered(d,mean,p):
        c=torch.empty_like(d);pc=torch.empty_like(d);pcc=torch.empty_like(d)
        _centered[(triton.cdiv(d.numel(),1024),)](d,mean,p,c,pc,pcc,d.numel(),d.shape[1],1024,num_warps=4,enable_fp_fusion=False)
        return c,pc,pcc
    @staticmethod
    def tilt(lp,t,c):
        out=torch.empty_like(lp)
        _tilted[(triton.cdiv(lp.numel(),1024),)](lp,t,c,out,lp.numel(),lp.shape[1],1024,num_warps=4,enable_fp_fusion=False)
        return out
    @staticmethod
    def finish(raw,rm,logits,lm,active):
        out=torch.empty_like(raw)
        _finish[(triton.cdiv(raw.numel(),1024),)](raw,rm,logits,lm,active,out,raw.numel(),raw.shape[1],1024,num_warps=4,enable_fp_fusion=False)
        return out
    @staticmethod
    def advance(point,kl,derivative,target,lower,lower_kl,upper,tiny,shrink):
        out=[torch.empty_like(point) for _ in range(4)]
        _advance[(triton.cdiv(point.numel(),128),)](point,kl,derivative,target,lower,lower_kl,upper,*out,
          point.numel(),tiny,shrink,128,num_warps=4,enable_fp_fusion=False)
        return tuple(out)

def bits_equal(a,b):
    return (a.shape==b.shape and a.dtype==b.dtype and
      torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),b.detach().contiguous().reshape(-1).view(torch.uint8)))
