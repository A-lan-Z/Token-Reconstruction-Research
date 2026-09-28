"""Bounded native component engine. No source truth is accepted or read."""
import sys, time, math
from pathlib import Path
from collections import Counter, defaultdict
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts/agent4_rescue'))
from shared import torch, sync
from executor import CurrentToken, TrialCache

class Window:
    """Reuse singleton kernels, retaining graph only between provisional tokens."""
    def __init__(self,ex): self.ex=ex
    def __call__(self,z):
        ex=self.ex; p=ex.prefix
        if tuple(t._version for t in p.parameters())!=ex.versions:
            raise RuntimeError('prefix changed')
        state=ex.state; ys=[]
        for j in range(len(z)):
            hidden=z[j].to(p.embed_tokens.weight.dtype).reshape(1,1,-1)
            positions=torch.full((1,1),ex.length+j,device=z.device,dtype=torch.long)
            rotary=p.rotary_emb(hidden,positions); trial=TrialCache(state)
            for layer in p.layers:
                hidden=p._hidden(layer(hidden,attention_mask=None,position_ids=positions,
                    position_embeddings=rotary,use_cache=False,**{p.cache_keyword:trial}))
            state=trial.new  # Never detach a provisional token's contribution.
            ys.append(hidden[0,0])
        return torch.stack(ys)

def losses(y,h):
    return .5*(y.float()-h.float()).square().sum(-1)/h.float().square().sum(-1).clamp_min(1e-8)

def gradient(win,z,h,mode):
    z=z.detach().float().clone().requires_grad_(); y=win(z); ls=losses(y,h)
    if mode=='diagonal':
        # Own-observation derivative for each slot; later rows cannot propose root.
        parts=[torch.autograd.grad(ls[j],z,retain_graph=j<len(z)-1)[0][j] for j in range(len(z))]
        g=torch.stack(parts)
    else:g=torch.autograd.grad(ls.sum(),z)[0]
    if not torch.isfinite(g).all():raise RuntimeError('nonfinite gradient')
    return g.detach()

class Vocabulary:
    def __init__(self,table):
        self.table=table; self.E=table.float(); self.norms=self.E.square().sum(-1)
        # Exact declared synthetic radius, fixed before native correctness scores.
        self.radius=.75*self.norms.sqrt().median()
    def rank(self,z,g,k):
        center=z.float()-self.radius*g.float()/g.float().norm().clamp_min(1e-12)
        d=self.norms-2*(self.E@center)+center.square().sum()
        idx=torch.topk(d,max(32,k),largest=False).indices
        # Native shortlist direct-distance convention; no incumbent exclusion.
        exact=(self.E[idx]-center).square().sum(-1)
        return [v for _,v in sorted(zip(exact.tolist(),idx.tolist()))[:k]]

class BudgetEnd(Exception):pass
class Meter:
    def __init__(self,rates,work_cap,wall_cap):
        self.rates=rates; self.cap=work_cap; self.wall_cap=wall_cap
        self.work=0.; self.count=Counter(); self.phase=defaultdict(float)
        sync(); self.start=time.perf_counter()
    def run(self,key,fn):
        sync()
        if self.work+self.rates[key]>self.cap+1e-12 or time.perf_counter()-self.start>=self.wall_cap:
            raise BudgetEnd
        start=time.perf_counter(); result=fn(); sync()
        self.phase[key]+=time.perf_counter()-start
        self.work+=self.rates[key]; self.count[key]+=1
        return result
    def summary(self):
        sync(); wall=time.perf_counter()-self.start
        return {'calibrated_work_seconds':self.work,'work_cap_seconds':self.cap,
                'actual_search_seconds':wall,'wall_cap_seconds':self.wall_cap,
                'wall_cap_met':wall<=self.wall_cap,'counts':dict(self.count),'phase_seconds':dict(self.phase)}

def search(win,vocab,h,initial,mode,rates,work_cap,wall_cap):
    width=1 if mode=='sequential' else len(initial)
    initial=tuple(initial[:width]); h=h[:width]; beam_size=1 if mode=='sequential' else 2
    proposal_width=16 if mode=='sequential' else 2
    table=vocab.table; meter=Meter(rates,work_cap,wall_cap)
    checked={}; history=[]; proposed_roots=set(); verified_roots=set(); accepted=False
    beams=[initial]; reason='four_round_cap'
    def verify(row):
        if row in checked:return
        def f():
            with torch.no_grad():
                y=win(table[list(row)]); value=float(losses(y,h).sum())
                strict=torch.allclose(y.float(),h.float(),atol=1e-5,rtol=1e-5)
            return value,bool(strict)
        value,strict=meter.run('forward'+str(width),f)
        if not math.isfinite(value):raise RuntimeError('nonfinite verification')
        checked[row]={'loss':value,'strict_allclose':strict}
        verified_roots.add(row[0])
    try:
        verify(initial)
        for rd in range(4):
            if any(checked[b]['strict_allclose'] for b in beams):
                accepted=True;reason='strict_residual';break
            candidates=set(beams); event={'round':rd,'starting_beams':[list(b) for b in beams],'proposals':[]}
            history.append(event)
            for b in beams:
                key='gradient_'+('local' if mode=='sequential' else mode)+str(width)
                g=meter.run(key,lambda:gradient(win,table[list(b)],h,mode))
                prop=[]
                for j in range(width):
                    ids=meter.run('scan'+str(proposal_width),lambda j=j:vocab.rank(table[b[j]],g[j],proposal_width))
                    prop.append(ids)
                    if j==0:proposed_roots.update(ids)
                    event['proposals'].append({'beam':list(b),'slot':j,'ids':ids})
                    for v in ids:
                        row=list(b);row[j]=v;candidates.add(tuple(row))
                if width>1:candidates.add(tuple(p[0] for p in prop))
            for row in sorted(candidates):verify(row)
            beams=sorted(candidates,key=lambda b:(checked[b]['loss'],b))[:beam_size]
            event['ending_beams']=[list(b) for b in beams]
            event['best_loss']=checked[beams[0]]['loss']
            if beams==[tuple(b) for b in event['starting_beams']]:
                reason='unchanged_beam';break
    except BudgetEnd:reason='shared_budget'
    # Even a partial final round must return the best actually checked hypothesis.
    best=min(checked,key=lambda b:(checked[b]['loss'],b)) if checked else None
    summary=meter.summary()
    counts=summary['counts']
    fwd=sum(n*(1 if k.endswith('1') else 3) for k,n in counts.items() if k.startswith(('forward','gradient')))
    reverse=sum(n*(1 if k.endswith('1') else 3)*(3 if 'diagonal' in k else 1) for k,n in counts.items() if k.startswith('gradient'))
    return {'mode':mode,'initial':list(initial),'returned_block':list(best) if best else None,
            'returned_loss':checked[best]['loss'] if best else None,
            'returned_strict_allclose':checked[best]['strict_allclose'] if best else False,
            'root_proposals':sorted(proposed_roots),'root_verified':sorted(verified_roots),
            'checks':[{'block':list(b),**v} for b,v in checked.items()],
            'rounds':history,'stop_reason':reason,'budget':summary,
            'forward_positions':fwd,'reverse_position_upper_count':reverse,
            'vocabulary_rows_scored':sum(n for k,n in counts.items() if k.startswith('scan'))*len(table),
            'committed_new_tokens':0,'truth_read':False}
