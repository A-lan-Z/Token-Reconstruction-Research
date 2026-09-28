"""Fitting-free joint-activation search on synthetic causal maps.

No pretrained weights, target dataset, learned inverse, or target token hints.
The tiny random transformer is a mechanism stress test, NOT a Llama surrogate
validated for prediction of real-world performance. All search returns are real
vocabulary tokens, verified through the supplied frozen forward function.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable
import math, time
import torch
import torch.nn.functional as F

DTYPE=torch.float64

@dataclass
class Work:
    forward_calls: int=0
    forward_sequences: int=0
    forward_positions: int=0
    reverse_calls: int=0
    reverse_vectors: int=0
    reverse_positions: int=0
    vocabulary_rows_scored: int=0
    # Matmul-based gradients/derivatives have other overhead not priced here.
    def add_forward(self,b:int,w:int):
        self.forward_calls+=1; self.forward_sequences+=b; self.forward_positions+=b*w
    @property
    def weighted_positions(self):
        # Merely a sensitivity-accounting proxy: backward = two forward passes.
        return self.forward_positions+2*self.reverse_positions

class TinyPrefix:
    """Four random pre-norm causal attention/MLP blocks; no learned training.

    Exact cached current-window forwards for its own declared FP64 mathematics.
    No RoPE, RMSNorm, grouped-query attention or BF16: not a Llama implementation.
    """
    def __init__(self,seed:int,vocab:int=128,d:int=32,layers:int=4,heads:int=4):
        if d%heads: raise ValueError('width must divide heads')
        g=torch.Generator().manual_seed(seed)
        self.E=torch.randn(vocab,d,generator=g,dtype=DTYPE)*0.5
        self.pos=torch.randn(64,d,generator=g,dtype=DTYPE)*0.04
        self.layers=[]; self.d=d; self.heads=heads
        for _ in range(layers):
            self.layers.append({
                'q':torch.randn(d,d,generator=g,dtype=DTYPE)/math.sqrt(d),
                'k':torch.randn(d,d,generator=g,dtype=DTYPE)/math.sqrt(d),
                'v':torch.randn(d,d,generator=g,dtype=DTYPE)/math.sqrt(d),
                'o':torch.randn(d,d,generator=g,dtype=DTYPE)/math.sqrt(d),
                'down':torch.randn(d,2*d,generator=g,dtype=DTYPE)/math.sqrt(d),
                'up':torch.randn(2*d,d,generator=g,dtype=DTYPE)/math.sqrt(2*d),
            })
    def empty_cache(self): return tuple((torch.empty(0,self.d,dtype=DTYPE),torch.empty(0,self.d,dtype=DTYPE)) for _ in self.layers)
    def forward(self,z:torch.Tensor,cache=None,return_cache=False):
        if z.ndim!=3 or z.shape[-1]!=self.d: raise ValueError('expected B,W,D')
        cache=self.empty_cache() if cache is None else cache
        n=cache[0][0].shape[0]; b,w,d=z.shape
        if n+w>self.pos.shape[0]: raise ValueError('position budget exceeded')
        x=z+self.pos[n:n+w]
        next_cache=[]
        for p,(kc,vc) in zip(self.layers,cache):
            u=F.layer_norm(x,(d,))
            q=u@p['q']; k=u@p['k']; v=u@p['v']
            K=torch.cat((kc.unsqueeze(0).expand(b,-1,-1),k),1)
            V=torch.cat((vc.unsqueeze(0).expand(b,-1,-1),v),1)
            Qh=q.view(b,w,self.heads,d//self.heads).transpose(1,2)
            Kh=K.view(b,n+w,self.heads,d//self.heads).transpose(1,2)
            Vh=V.view(b,n+w,self.heads,d//self.heads).transpose(1,2)
            s=Qh@Kh.transpose(-1,-2)/math.sqrt(d//self.heads)
            mask=torch.arange(n+w)[None,:]<=n+torch.arange(w)[:,None]
            a=torch.softmax(s.masked_fill(~mask[None,None],-torch.inf),-1)
            o=(a@Vh).transpose(1,2).reshape(b,w,d)@p['o']
            x=x+0.4*o
            u=F.layer_norm(x,(d,))
            x=x+0.4*(F.gelu(u@p['down'])@p['up'])
            next_cache.append((K[0].detach().clone(),V[0].detach().clone()))
        return (x,tuple(next_cache)) if return_cache else x
    def commit(self,ids:torch.Tensor,cache):
        with torch.no_grad(): return self.forward(self.E[ids][None],cache,True)[1]


def stable_nearest(E:torch.Tensor,z:torch.Tensor,k:int):
    # Distances are exact formula, but finite-precision ties use ascending token ID.
    scores=E.square().sum(-1)[None]-2*z@E.T
    return torch.argsort(scores,dim=-1,stable=True)[...,:k]


def observed_losses(y:torch.Tensor,h:torch.Tensor):
    # Fixed observed scale for all candidates; not truth- or prediction-dependent.
    # Scale per position avoids large-norm positions dominating merely by amplitude.
    scales=h.square().sum(-1).clamp_min(1e-8)
    return 0.5*(y-h[None]).square().sum(-1)/scales[None]


def proposals(E:torch.Tensor,z:torch.Tensor,g:torch.Tensor,p:int):
    # Fixed proximal radius: gradient step at most 0.75*median embedding norm.
    # Scale selected before independent model seeds, not fit from true token labels.
    radius=0.75*torch.median(torch.linalg.vector_norm(E,dim=-1))
    gn=torch.linalg.vector_norm(g,dim=-1,keepdim=True)
    step=radius*g/gn.clamp_min(1e-12)
    return stable_nearest(E,(z-step).reshape(-1,E.shape[-1]),p).reshape(*z.shape[:2],p)


def window_search(model:TinyPrefix,obs:torch.Tensor,mode:str='joint',window:int=3,
                  beam:int=2,p:int=2,rounds:int=4, fast:bool=False, block_commit:bool=False) -> dict:
    """Receding-horizon discrete search. obs excludes known BOS.

    mode joint: proposal gradients of ALL visible block losses.
    mode diagonal: identical block verification, but each proposal uses only
      the same-position output loss. For diagnostic parity BOTH modes obtain
      the full loss Jacobian using W reverse vectors; this is NOT the optimized
      one-reverse implementation possible for joint.
    mode sequential: window=1, beam=1 and one reverse vector.
    """
    if mode not in ('joint','diagonal','sequential'): raise ValueError(mode)
    if mode=='sequential': window=1; beam=1
    E=model.E; n,d=obs.shape; work=Work(); t0=time.perf_counter()
    cache=model.empty_cache(); cache=model.commit(torch.tensor([0]),cache); work.add_forward(1,1)
    committed=[]; carry=None; traces=[]
    with torch.no_grad(): raw=stable_nearest(E,obs,1).squeeze(-1)
    work.vocabulary_rows_scored+=n*len(E)
    i=0
    while i<n:
        w=min(window,n-i); h=obs[i:i+w]
        row=raw[i:i+w].clone()
        if carry is not None:
            m=min(len(carry),w); row[:m]=carry[:m]
        beams=row[None]
        for r in range(rounds):
            z=E[beams].detach().requires_grad_(True)
            y=model.forward(z,cache); work.add_forward(len(beams),w)
            ls=observed_losses(y,h)
            if fast and float(ls.detach().sum(-1).min())<1e-20:
                ix=int(torch.argmin(ls.detach().sum(-1)))
                beams=beams[ix:ix+1]
                best=float(ls.detach().sum(-1)[ix])
                traces.append({'position':i,'round':r,'width':w,'candidates':0,'best_loss':best,
                               'root_candidates':beams[:,0].tolist(),'early_match':True})
                break
            # The full joint gradient needs only ONE reverse evaluation.
            if fast and mode!='diagonal':
                grad=torch.autograd.grad(ls.sum(),z)[0]
                work.reverse_calls+=1; work.reverse_vectors+=1
                work.reverse_positions+=len(beams)*w
            else:
                grad=None
            vjps=[]
            for j in range(w) if grad is None else []:
                vjps.append(torch.autograd.grad(ls[:,j].sum(),z,retain_graph=j<w-1)[0])
            if vjps:
                work.reverse_calls+=w; work.reverse_vectors+=w
                work.reverse_positions+=len(beams)*w*w
            if mode=='diagonal':
                grad=torch.stack([vjps[j][:,j] for j in range(w)],dim=1)
            elif vjps:
                grad=torch.stack(vjps).sum(0)
            with torch.no_grad():
                prop=proposals(E,z.detach(),grad.detach(),p)
                work.vocabulary_rows_scored+=len(beams)*w*len(E)
                candidates={tuple(x) for x in beams.tolist()}
                for b in range(len(beams)):
                    for j in range(w):
                        for v in prop[b,j].tolist():
                            vrow=beams[b].clone(); vrow[j]=v; candidates.add(tuple(vrow.tolist()))
                    # One simultaneous, Jacobi-style proposal, still verified.
                    candidates.add(tuple(prop[b,:,0].tolist()))
                cand=torch.tensor(sorted(candidates),dtype=torch.long)
                yc=model.forward(E[cand],cache); work.add_forward(len(cand),w)
                losses=observed_losses(yc,h).sum(-1)
                order=torch.argsort(losses,stable=True)[:beam]
                new_beams=cand[order]
                best=float(losses[order[0]])
                traces.append({'position':i,'round':r,'width':w,'candidates':len(cand),'best_loss':best,'root_candidates':sorted(set(cand[:,0].tolist())),'root_proposals':sorted(set(prop[:,0].reshape(-1).tolist()))})
                # Exact match in this matched FP64 toy is a genuine stopping case;
                # real target mismatch cannot use it as universal certificate.
                same=torch.equal(new_beams,beams)
                beams=new_beams
                if best<1e-20 or same: break
        take=w if block_commit and best<1e-20 else 1
        chosen=beams[0,:take].clone(); committed.extend(chosen.tolist())
        carry=beams[0,take:].clone() if w>take else None
        cache=model.commit(chosen,cache); work.add_forward(1,take)
        i+=take
    return {'prediction':committed,'work':asdict(work)|{'weighted_positions':work.weighted_positions},
            'seconds':time.perf_counter()-t0,'trace':traces}
