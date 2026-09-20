"""All-vocabulary scalar KL and derivative reductions; no support pruning."""
import torch
import triton
import triton.language as tl

@triton.jit
def _tiles(L,C,T,MAX,SUM,WEIGHT,V:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);tile=tl.program_id(1)
    ix=tile*B+tl.arange(0,B);valid=ix<V
    lp=tl.load(L+row*V+ix,valid,float("-inf"))
    centered=tl.load(C+row*V+ix,valid,0.)
    step=tl.load(T+row)
    value=lp-step*centered
    maximum=tl.max(value,0)
    weight=tl.exp(value-maximum)
    total=tl.sum(weight,0);weighted=tl.sum(weight*centered,0)
    offset=row*N+tile
    tl.store(MAX+offset,maximum);tl.store(SUM+offset,total);tl.store(WEIGHT+offset,weighted)

@triton.jit
def _merge(MAX,SUM,WEIGHT,T,BASE,CORRECTION,KL,DERIV,N:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);ix=tl.arange(0,B);valid=ix<N
    m=tl.load(MAX+row*N+ix,valid,float("-inf"))
    z=tl.load(SUM+row*N+ix,valid,0.)
    w=tl.load(WEIGHT+row*N+ix,valid,0.)
    maximum=tl.max(m,0);scale=tl.exp(m-maximum)
    total=tl.sum(z*scale,0);weighted=tl.sum(w*scale,0)
    logz=maximum+tl.log(total)
    step=tl.load(T+row);base=tl.load(BASE+row);correction=tl.load(CORRECTION+row)
    tl.store(KL+row,logz-base+step*correction)
    tl.store(DERIV+row,correction-weighted/total)

class TiledEvaluator:
    def __init__(self,logp,centered,base,correction):
        if logp.dtype!=torch.float32 or not logp.is_cuda or not logp.is_contiguous() or not centered.is_contiguous():raise ValueError("FP32contiguous CUDA matrices required")
        self.logp=logp;self.centered=centered;self.base=base;self.correction=correction
        self.rows,self.vocab=logp.shape;self.tiles=triton.cdiv(self.vocab,1024)
        self.work=torch.empty((3,self.rows,self.tiles),device=logp.device,dtype=logp.dtype)
    def __call__(self,step):
        kl=torch.empty_like(self.base);deriv=torch.empty_like(self.base)
        _tiles[(self.rows,self.tiles)](self.logp,self.centered,step,self.work[0],self.work[1],self.work[2],
          self.vocab,self.tiles,1024,num_warps=4,enable_fp_fusion=False)
        _merge[(self.rows,)](self.work[0],self.work[1],self.work[2],step,self.base,self.correction,kl,deriv,
          self.tiles,triton.next_power_of_2(self.tiles),num_warps=4,enable_fp_fusion=False)
        return kl,deriv
