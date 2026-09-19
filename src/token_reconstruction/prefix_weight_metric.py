"""Prefix-native proposal geometry with no independently fitted parameters.

The transform is calculated exclusively from residual-write matrices of the
supplied prefix. Its cached transformed embedding table is disposable and is
invalidated by prefix parameter replacement or in-place updates.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

class PrefixWeightMetric:
    def __init__(self,prefix):
        if any(p.requires_grad for p in prefix.parameters()):
            raise ValueError("freeze the supplied prefix before constructing proposal state")
        self.prefix=prefix
        self.signature=self._signature()
        self.transform=None
        self.table=None

    def _signature(self):
        return tuple((n,id(p),p.data_ptr(),p._version) for n,p in self.prefix.named_parameters())

    def _check(self):
        if self.signature!=self._signature():
            raise RuntimeError("prefix changed: rebuild the derived proposal cache")

    @torch.no_grad()
    def build(self,chunk=4096):
        self._check()
        weight=self.prefix.embed_tokens.weight
        d=weight.shape[-1]
        C=torch.zeros((d,d),device=weight.device,dtype=torch.float32)
        for layer in self.prefix.layers:
            for module in (layer.self_attn.o_proj,layer.mlp.down_proj):
                W=module.weight.detach().float()
                G=W@W.T
                scale=G.trace()
                if not torch.isfinite(scale) or scale<=0:raise ValueError("invalid residual-write matrix")
                C+=G/scale
        values,U=torch.linalg.eigh(C)
        floor=torch.finfo(values.dtype).eps*values.max()
        self.transform=U*values.clamp_min(floor).rsqrt()
        self.table=torch.empty(weight.shape,device=weight.device,dtype=torch.float32)
        for lo in range(0,len(weight),chunk):
            self.table[lo:lo+chunk]=F.normalize(weight[lo:lo+chunk].float()@self.transform,dim=-1)
        self._check()
        return {'min_eigenvalue':float(values.min()),'max_eigenvalue':float(values.max()),
                'transform_bytes':self.transform.numel()*self.transform.element_size(),
                'embedding_cache_bytes':self.table.numel()*self.table.element_size(),
                'fit_steps':0,'fit_examples':0}

    @torch.no_grad()
    def propose(self,observations,k=64):
        self._check()
        if self.table is None:raise RuntimeError("build the cache first")
        if observations.ndim!=2 or observations.shape[1]!=self.table.shape[1]:raise ValueError("expected [positions,width]")
        if not torch.isfinite(observations).all():raise ValueError("nonfinite observations")
        if not 1<=k<=len(self.table):raise ValueError("invalid candidate budget")
        q=F.normalize(observations.to(self.table.device).float()@self.transform,dim=-1)
        return (q@self.table.T).topk(k,dim=-1).indices
