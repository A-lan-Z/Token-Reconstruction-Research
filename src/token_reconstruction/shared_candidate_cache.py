"""Numerically identical native A2 with a disposable shared-context cache."""
import torch
from token_reconstruction.public_prefix import PublicPrefixCache

class SharedContext:
    def __init__(self, committed, width, layers):
        self.committed=committed
        self.width=width
        self.updated=[False]*layers
        self.length=committed.length
        if self.length<1:raise ValueError("known BOS context required")
    def get_seq_length(self, layer_idx=0):
        return self.length+int(self.updated[layer_idx])
    def update(self, key_states,value_states,layer_idx,cache_kwargs=None):
        if self.updated[layer_idx]:raise ValueError("candidate cache may update each layer once")
        layer=self.committed.backend.layers[layer_idx]
        keys,values=layer.keys,layer.values
        expected=(self.width,keys.shape[1],1,keys.shape[3])
        if tuple(key_states.shape)!=expected or tuple(value_states.shape)!=expected:
            raise ValueError("candidate shape changed")
        if keys.shape[0]!=1 or keys.shape[2]!=self.length:
            raise ValueError("expected one committed context")
        self.updated[layer_idx]=True
        # Same contiguous final tensors as native repeat_interleave + cat.
        # expand itself is a read-only view; torch.cat materializes the result.
        return (torch.cat([keys.expand(self.width,-1,-1,-1),key_states],dim=-2),
                torch.cat([values.expand(self.width,-1,-1,-1),value_states],dim=-2))

@torch.no_grad()
def shared_candidates(prefix,cache,ids,pos):
    if ids.ndim!=1 or pos!=cache.length or len(ids)>4096:
        raise ValueError("unsupported candidate request")
    temporary=PublicPrefixCache(SharedContext(cache,len(ids),len(prefix.layers)),pos)
    return prefix.run_cached(ids.reshape(-1,1),temporary,pos)[:,-1].float()

