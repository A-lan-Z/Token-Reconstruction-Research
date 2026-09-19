"""Frozen candidate-only family; no truth imports until every array is frozen."""
from pathlib import Path
import sys,json,time,hashlib,argparse,statistics,resource
from collections import Counter,defaultdict
R=Path(__file__).resolve().parents[2];P=R.parent/'TRR-0016';X=R/'experiments/TRR-0018';OUT=R/'outputs/TRR-0018/development'
sys.path.insert(0,str(R/'scripts/trr0014'))
import native as n
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from token_reconstruction.prefix_fragments import PrefixFragmentProposer,expand_candidates
from token_reconstruction.prefix_fragment_frequency import frequency_candidates
VARIANTS=tuple(f'{policy}{parents}' for parents in (64,128) for policy in ('frequency','frequency_score','score','rank_vote','mix64'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)
def bind():
    paths=[Path(__file__),R/'src/token_reconstruction/prefix_fragments.py',R/'src/token_reconstruction/prefix_weight_metric.py',R/'src/token_reconstruction/prefix_fragment_frequency.py',X/'PLAN.md',P/'outputs/TRR-0016/observations.safetensors',P/'outputs/TRR-0016/metadata.json',n.ASSETS/'backup/prefix.safetensors',R.parent/'TRR-0014/outputs/TRR-0014/tokenizer_suffix_cache.json']
    return {str(p):sha(p) for p in paths}
def suffix_order(parents,cache,base):
    seen=set(base);fs=[cache[p] for p in parents];order={}
    for j in range(max(map(len,fs),default=0)):
        for f in fs:
            if j<len(f) and f[j] not in seen and f[j] not in order:order[f[j]]=len(order)
    return order
@torch.no_grad()
def propose(proposer,h):
    proposer.metric._check()
    q=F.normalize(h.to('cuda').float()@proposer.metric.transform,dim=-1)
    a=q@proposer.metric.table.T;b=q@proposer.dictionary.T
    first=a.topk(64,dim=-1).indices;second=b.topk(64,dim=-1).indices
    base=torch.cat([first,second],-1).cpu().tolist()
    broad_a=a.topk(128,dim=-1).indices.cpu().tolist();broad_b=b.topk(128,dim=-1).indices.cpu().tolist()
    details=[];allfragments=[]
    for original,aa,bb in zip(base,broad_a,broad_b):
        unique=list(dict.fromkeys(original));by_n={}
        for width in (64,128):
            parents=unique if width==64 else list(dict.fromkeys(original+aa+bb))
            order=suffix_order(parents,proposer.suffix_cache,unique)
            counts=Counter(v for p in parents for v in proposer.suffix_cache[p])
            ranks={}
            for table in (aa[:width],bb[:width]):
                for rank,p in enumerate(table):ranks[p]=min(ranks.get(p,10**9),rank)
            weights=defaultdict(float)
            for p in parents:
                for v in proposer.suffix_cache[p]:weights[v]+=1/(64+ranks[p])
            by_n[width]=(order,counts,weights)
        union=list(by_n[128][0]);allfragments.append(union);details.append((unique,by_n))
    width=max(map(len,allfragments),default=1)
    ids=torch.tensor([v+[0]*(width-len(v)) for v in allfragments],device='cuda')
    scores=torch.maximum(a.gather(1,ids),b.gather(1,ids)).cpu().tolist()
    del a,b,ids
    output={v:[] for v in VARIANTS};output.update(fragment256=[],fragment512=[])
    for original,fragments,values,(base_ids,by_n) in zip(base,allfragments,scores,details):
        direct=dict(zip(fragments,values));slots=256-len(base_ids)
        for width,(order,counts,weights) in by_n.items():
            by_vote=sorted(order,key=lambda v:(-counts[v],order[v],v))
            by_score=sorted(order,key=lambda v:(-direct[v],order[v],v))
            orders={'frequency':by_vote,'frequency_score':sorted(order,key=lambda v:(-counts[v],-direct[v],order[v],v)),
                    'score':by_score,'rank_vote':sorted(order,key=lambda v:(-weights[v],order[v],v)),
                    'mix64':list(dict.fromkeys(by_vote[:64]+by_score))}
            for policy,ranked in orders.items():
                result=base_ids+ranked[:slots];result += [result[0]]*(256-len(result));output[f'{policy}{width}'].append(result)
        for cap in (256,512):output[f'fragment{cap}'].append(expand_candidates(original,proposer.suffix_cache,cap))
    return {v:torch.tensor(ids,dtype=torch.int64) for v,ids in output.items()},torch.tensor(base)
def setup():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=n.environment();prefix=n.load_prefix()
    cache=json.loads((R.parent/'TRR-0014/outputs/TRR-0014/tokenizer_suffix_cache.json').read_text())
    n.sync();start=time.perf_counter();proposer=PrefixFragmentProposer(prefix,cache);stats=proposer.build(256);n.sync()
    return prefix,proposer,{'environment':env,'cache_build_seconds':time.perf_counter()-start,'cache':stats}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualify',action='store_true');args=parser.parse_args()
    binding=bind();prefix,proposer,preparation=setup();OUT.mkdir(parents=True,exist_ok=True)
    if args.qualify:
        ids=torch.tensor([[128000]+list(range(1000,1127))],device='cuda')
        with torch.no_grad():h=prefix.forward_full(ids)[0].cpu()
        n.sync();t=time.perf_counter();data,base=propose(proposer,h);n.sync();elapsed=time.perf_counter()-t
        assert torch.equal(base,proposer.base(h).cpu())
        assert torch.equal(data['frequency64'],torch.tensor([frequency_candidates(v,proposer.suffix_cache) for v in base.tolist()]))
        assert torch.equal(data['fragment256'],proposer.propose(h,budget=256).cpu())
        assert torch.equal(data['fragment512'],proposer.propose(h,budget=512).cpu())
        for v in data:assert data[v].shape==(128,512 if v=='fragment512' else 256)
        n.guard();write(X/'development_qualification.json',{'binding':binding,'preparation':preparation,'seconds_all12_rules':elapsed,'exact_native_base_frequency_fragments':True,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'utc':time.time()});return
    assert json.loads((X/'development_qualification.json').read_text())['binding']==binding
    rows=json.loads((P/'outputs/TRR-0016/metadata.json').read_text());obs=load_file(str(P/'outputs/TRR-0016/observations.safetensors'))
    entries=[]
    for i,row in enumerate(rows):
        key=row['id'];path=OUT/(key+'.safetensors');receipt=OUT/(key+'.json')
        if receipt.exists():
            e=json.loads(receipt.read_text());assert e['binding']==binding and sha(path)==e['sha256'];entries.append(e);continue
        assert not path.exists();n.guard();n.sync();t=time.perf_counter();data,base=propose(proposer,obs[key]);n.sync();elapsed=time.perf_counter()-t
        for current,old in [('frequency64','frequency256'),('fragment256','fragment256'),('fragment512','fragment512')]:
            control=load_file(str(P/'outputs/TRR-0016/predictions'/(key+'__'+old+'.safetensors')))['candidates']
            assert torch.equal(data[current][1:],control[1:]),(key,current)
        save_file(data,str(path));e={**row,'path':str(path.relative_to(R)),'sha256':sha(path),'binding':binding,'seconds_all12_rules':elapsed,'frozen_unix':time.time(),'anchors_identical':True};write(receipt,e);entries.append(e)
        if (i+1)%16==0:print(i+1,'/',len(rows),'candidate arrays frozen',flush=True)
    assert bind()==binding
    write(X/'development_freeze.json',{'task_id':'TRR-0018','variants':list(VARIANTS)+['fragment256','fragment512'],'binding':binding,'entries':entries,'preparation':preparation,'truth_read':False,'finished_unix':time.time(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_host_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    print('Full candidate matrix frozen',flush=True)
if __name__=='__main__':main()
