"""Parallel same-prefix context subtraction, opened-panel proposal experiment."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
env=environment();prefix=load_prefix();metric=PrefixWeightMetric(prefix);metric.build()
V,D=prefix.embed_tokens.weight.shape
intrinsic=torch.empty_like(prefix.embed_tokens.weight)
table=torch.empty_like(metric.table)
sync();start=time.perf_counter()
for lo in range(0,V,256):
    if lo%8192==0:guard()
    h=prefix.embed_tokens.weight[lo:lo+256].unsqueeze(1)
    for layer in prefix.layers:h=h+layer.mlp(layer.post_attention_layernorm(h))
    intrinsic[lo:lo+len(h)]=h[:,0]
    table[lo:lo+len(h)]=F.normalize(h[:,0].float()@metric.transform,dim=-1)
sync();setup=time.perf_counter()-start;guard()
def rank(q,k):return (F.normalize(q.float()@metric.transform,dim=-1)@table.T).topk(k,dim=-1).indices
@torch.no_grad()
def proposal(h):
    h=h.to('cuda').float()
    initial=metric.propose(h,32);other=rank(h,32)
    allids=[initial,other];branches={}
    for seed,ids in [('embedding',initial),('intrinsic',other)]:
        guess=ids[:,0].clone();guess[0]=128000;hist=[]
        for step in range(4):
            response=prefix.forward_full(guess.unsqueeze(0))[0].float()
            query=h-response+intrinsic[guess].float()
            ids=rank(query,16);hist.append(ids)
            guess=ids[:,0].clone();guess[0]=128000
        allids.extend(hist);branches[seed]=torch.cat([initial,other]+hist,dim=-1)
    branches['union']=torch.cat(allids,dim=-1)
    return branches
fixture=torch.tensor([[128000]+list(range(1000,1127))],device='cuda')
proposal(prefix.forward_full(fixture)[0]);guard()
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
dst=OUT/'context_dev_r1';dst.mkdir(exist_ok=False);entries=[]
for row in meta:
    sync();start=time.perf_counter();branches=proposal(obs[row['id']]);sync();seconds=time.perf_counter()-start
    for name,ids in branches.items():
        tokens=ids[:,0].clone();tokens[0]=128000
        path=dst/(row['id']+'__'+name+'.safetensors');save_file({'tokens':tokens.cpu(),'candidates':ids[1:].cpu()},str(path))
        entries.append({**row,'method':name,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':seconds})
    guard()
write(X/'context_dev_r1_freeze.json',{'environment':env,'entries':entries,'setup_seconds':setup,'observations_sha256':digest(panel/'observations.safetensors'),'truth_read':False,'scope':'opened-panel proposal recall only; token output is original metric top1, not reconstruction','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
print('All144 proposal outputs frozen',flush=True)
