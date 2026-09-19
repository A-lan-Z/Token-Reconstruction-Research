"""No-fit suffix consensus from the same prefix-derived lookup candidates."""
from collections import Counter
import torch


def frequency_candidates(base, suffix_cache, budget=256):
    """Keep base tokens, then rank fragments by number of distinct parents."""
    if not base or budget < len(base):
        raise ValueError("budget must accommodate the nonempty base")
    result=list(dict.fromkeys(int(v) for v in base));seen=set(result)
    counts=Counter(v for parent in result for v in suffix_cache[parent])
    fragments=[suffix_cache[int(v)] for v in base]
    order={}
    for depth in range(max((len(f) for f in fragments),default=0)):
        for fragment in fragments:
            if depth<len(fragment) and fragment[depth] not in seen and fragment[depth] not in order:
                order[fragment[depth]]=len(order)
    ranked=sorted(order,key=lambda v:(-counts[v],order[v],v))
    result.extend(ranked[:budget-len(result)])
    return result+[result[0]]*(budget-len(result))


class FrequencyFragmentView:
    """Reuse the original prefix tables and cache validation; fit nothing new."""
    def __init__(self,proposer):self.proposer=proposer

    @torch.no_grad()
    def propose(self,observations,fragments=True):
        base=self.proposer.base(observations)
        if not fragments:return base
        result=[frequency_candidates(row,self.proposer.suffix_cache) for row in base.cpu().tolist()]
        return torch.tensor(result,device=base.device,dtype=torch.long)