"""Prediction-only retrospective probe; no evaluation truth is imported."""
from native import *
from torch.nn import functional as F
import argparse,resource
a=argparse.ArgumentParser();a.add_argument('--name',default='lookup_dev_r1');args=a.parse_args()
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
env=environment();prefix=load_prefix();guard()
dst=X/args.name;dst.mkdir(exist_ok=False)
OUT.mkdir(parents=True,exist_ok=True)
obsfile=RESCUE/'development/observations.safetensors'
obs=load_file(str(obsfile));meta=json.loads((RESCUE/'development/metadata.json').read_text())
# Entire cache is generated from the current prefix, without public examples or fitting.
cache=new_context(prefix);V,D=prefix.embed_tokens.weight.shape
sync();t=time.perf_counter();table=torch.empty((V,D),device='cuda',dtype=torch.float32)
for lo in range(0,V,256):
    if lo%8192==0:guard()
    ids=torch.arange(lo,min(lo+256,V),device='cuda')
    table[lo:lo+len(ids)]=candidates(prefix,cache,ids,1).float()
sync();build=time.perf_counter()-t
norms=table.square().sum(-1);lengths=norms.sqrt()
save_file({'response':table.cpu()},str(OUT/(args.name+'_responses.safetensors')))
write(dst/'cache.json',{'environment':env,'build_seconds':build,'bytes':table.numel()*table.element_size(),'prefix_sha256':digest(ASSETS/'backup/prefix.safetensors'),'cache_sha256':digest(OUT/(args.name+'_responses.safetensors')),'batch_contract':256,'reference_context':[128000],'position':1,'fit_steps':0})
def rank(query,k,kind):
    if kind=='cos':scores=(table@query)/lengths.clamp_min(1e-8);return scores.topk(k).indices
    scores=norms-2*(table@query);return scores.topk(k,largest=False).indices
def decode(h,method):
    cache=new_context(prefix);tokens=[128000];trace=[]
    sync();begin=time.perf_counter()
    for pos in range(1,len(h)):
        target=h[pos].to('cuda').float();query=target;seen=set();checks=[];best=None;bestloss=float('inf')
        rounds=3 if method=='transport' else 1;k=16 if method=='transport' else 64
        for n in range(rounds):
            ids=rank(query,k,'cos' if method=='cosine' else 'mse')
            output=candidates(prefix,cache,ids,pos).float()
            losses=(output-target).square().mean(-1);j=int(losses.argmin())
            for v,l in zip(ids.tolist(),losses.tolist()):checks.append({'token':v,'mse':l,'round':n})
            if float(losses[j])<bestloss:best,bestloss=int(ids[j]),float(losses[j])
            # Re-express target in the BOS response coordinates of the current best proposal.
            # All operands depend only on the observation, recovered history and current prefix.
            query=target-output[j]+table[ids[j]]
        tokens.append(best);prefix.run_cached(torch.tensor([[best]],device='cuda'),cache,pos)
        trace.append({'position':pos,'token':best,'mse':bestloss,'checks':checks})
    sync();return {'tokens':tokens,'trace':trace,'seconds':time.perf_counter()-begin,'method':method}
for method in ['euclidean','cosine','transport']:
    for row in meta:
        guard();r=decode(obs[row['id']],method);r['record_id']=row['id'];r['environment']=env
        write(dst/(method+'_'+row['id']+'.json'),r);print(method,row['id'],r['seconds'],flush=True)
# Comparator has an explicit separate preparation phase.
from comparator import prepare,decode as baseline
lens,emb=prepare(prefix)
for row in meta:
    guard();r=baseline(prefix,obs[row['id']],lens,emb);r['record_id']=row['id'];r['environment']=env
    write(dst/('a1a2_'+row['id']+'.json'),r);print('a1a2',row['id'],r['seconds'],flush=True)
write(dst/'freeze.json',{'environment':env,'observations_sha256':digest(obsfile),'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(dst.glob('*.json'))],'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'truth_read':False,'retrospective_development':True})
