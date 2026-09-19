"""Exploratory deterministic ordering policies; none is an active method yet."""
from collections import Counter
from .prefix_fragments import expand_candidates

VARIANTS=('shortest','ends','frequency','frequency64','interleave','shortest_interleave')

def ordered_fragments(base, cache, variant, budget=256):
    result=list(dict.fromkeys(int(v) for v in base));seen=set(result)
    parents=list(base)
    if variant in ('interleave','shortest_interleave'):
        parents=[v for pair in zip(base[:64],base[64:]) for v in pair]
    fragments=[list(cache[v]) for v in parents]
    if variant in ('shortest','shortest_interleave'): fragments=[v[::-1] for v in fragments]
    elif variant=='ends':
        fragments=[[v[i//2] if i%2==0 else v[-(i//2+1)] for i in range(len(v))] for v in fragments]
    if variant.startswith('frequency'):
        counts=Counter(v for b in dict.fromkeys(base) for v in cache[b])
        first={v:i for i,v in enumerate(expand_candidates(base,cache,4096)) if v not in seen}
        ranked=sorted(counts,key=lambda v:(-counts[v],first.get(v,4096),v))
        if variant=='frequency64': ranked=[v for v in ranked if v not in seen][:64]+expand_candidates(base,cache,budget)
        else:ranked=ranked+expand_candidates(base,cache,budget)
    else:
        ranked=[f[d] for d in range(max(map(len,fragments),default=0)) for f in fragments if d<len(f)]
    for v in ranked:
        if v not in seen:
            result.append(v);seen.add(v)
            if len(result)==budget:return result
    return result+[result[0]]*(budget-len(result))