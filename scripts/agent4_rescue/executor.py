"""Functional current-token Llama path with immutable detached committed K/V."""
from shared import *
class TrialCache:
    def __init__(self,old):
        self.old=old
        self.new={}
    def update(self,k,v,index,cache_kwargs=None):
        if index in self.old:
            a,b=self.old[index]
            k=torch.cat((a.expand(k.shape[0],-1,-1,-1),k),dim=-2)
            v=torch.cat((b.expand(v.shape[0],-1,-1,-1),v),dim=-2)
        self.new[index]=(k,v)
        return k,v

class CurrentToken:
    def __init__(self,prefix,bos=128000):
        self.prefix=prefix;self.state={};self.length=0
        self.versions=tuple(p._version for p in prefix.parameters())
        self.commit(bos)
    def invalidate(self):
        self.state={};self.length=0
        self.versions=tuple(p._version for p in self.prefix.parameters())
    def evaluate(self,z,commit=False):
        if tuple(p._version for p in self.prefix.parameters())!=self.versions:
            raise RuntimeError('prefix weights changed: invalidate/rebuild committed cache')
        hidden=z.to(self.prefix.embed_tokens.weight.dtype).reshape(-1,1,z.shape[-1])
        positions=torch.full((hidden.shape[0],1),self.length,device=z.device,dtype=torch.long)
        rotary=self.prefix.rotary_emb(hidden,positions)
        cache=TrialCache(self.state)
        for layer in self.prefix.layers:
            hidden=self.prefix._hidden(layer(hidden,attention_mask=None,position_ids=positions,
                position_embeddings=rotary,use_cache=False,**{self.prefix.cache_keyword:cache}))
        if commit:
            if hidden.shape[0]!=1:raise ValueError('only one token can commit')
            self.state={i:(k.detach(),v.detach()) for i,(k,v) in cache.new.items()}
            self.length+=1
        return hidden[:,0]
    def __call__(self,z):
        return self.evaluate(z).reshape(z.shape)
    def commit(self,token):
        with torch.no_grad():self.evaluate(self.prefix.embed_tokens.weight[token],commit=True)
