"""No-fit K256 proposals combining suffix consensus and native lookup scores."""
from collections import Counter
import torch
from torch.nn import functional as F

def fragment_pool(base, parents, suffix_cache):
    base=list(dict.fromkeys(base));parents=list(dict.fromkeys(parents));seen=set(base)
    suffixes=[suffix_cache[p] for p in parents]
    counts=Counter(v for values in suffixes for v in values)
    order={}
    for depth in range(max(map(len,suffixes),default=0)):
        for values in suffixes:
            if depth<len(values) and values[depth] not in seen and values[depth] not in order:
                order[values[depth]]=len(order)
    votes=sorted(order,key=lambda v:(-counts[v],order[v],v))
    return base,list(order),votes

def select_fragments(base,fragments,votes,scores):
    order={v:i for i,v in enumerate(fragments)}
    ranked=sorted(fragments,key=lambda v:(-scores[v],order[v],v))
    result=base+list(dict.fromkeys(votes[:64]+ranked))[:256-len(base)]
    return result+[result[0]]*(256-len(result))

class MixedFragmentView:
    """Preserve base64+64; use128+128 parents,64 voting slots, then similarity."""
    def __init__(self,proposer):self.proposer=proposer
    @torch.no_grad()
    def propose(self,observations,fragments=True):
        p=self.proposer;p.metric._check()
        if p.dictionary is None:raise RuntimeError('build prefix-derived tables first')
        if not fragments:return p.base(observations)
        if observations.ndim!=2 or observations.shape[1]!=p.dictionary.shape[1]:raise ValueError('expected positions by width')
        if not bool(torch.isfinite(observations).all()):raise ValueError('nonfinite observations')
        query=F.normalize(observations.to(p.dictionary.device).float()@p.metric.transform,dim=-1)
        embedding=query@p.metric.table.T;intrinsic=query@p.dictionary.T
        base=torch.cat([embedding.topk(64,dim=-1).indices,intrinsic.topk(64,dim=-1).indices],-1).cpu().tolist()
        a=embedding.topk(128,dim=-1).indices.cpu().tolist();b=intrinsic.topk(128,dim=-1).indices.cpu().tolist()
        pools=[fragment_pool(original,original+aa+bb,p.suffix_cache) for original,aa,bb in zip(base,a,b)]
        width=max(1,max((len(pool[1]) for pool in pools),default=0))
        ids=torch.tensor([values+[0]*(width-len(values)) for _,values,_ in pools],device=embedding.device,dtype=torch.long)
        scores=torch.maximum(embedding.gather(1,ids),intrinsic.gather(1,ids)).cpu().tolist()
        result=[select_fragments(base,values,votes,dict(zip(values,ss))) for (base,values,votes),ss in zip(pools,scores)]
        return torch.tensor(result,device=embedding.device,dtype=torch.long)
