"""Causal native verification with a prefix-intrinsic consistency correction."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
env=environment();prefix=load_prefix();metric=PrefixWeightMetric(prefix);metric.build()
V,D=prefix.embed_tokens.weight.shape
intrinsic=torch.empty_like(prefix.embed_tokens.weight);table=torch.empty_like(metric.table)
for lo in range(0,V,256):
    if lo%8192==0:guard()
    h=prefix.embed_tokens.weight[lo:lo+256].unsqueeze(1)
    for layer in prefix.layers:h=h+layer.mlp(layer.post_attention_layernorm(h))
    intrinsic[lo:lo+len(h)]=h[:,0];table[lo:lo+len(h)]=F.normalize(h[:,0].float()@metric.transform,dim=-1)
def rank(q,k):return (F.normalize(q.float()@metric.transform,dim=-1)@table.T).topk(k,dim=-1).indices
@torch.no_grad()
def decode(h):
    sync();start=time.perf_counter();h=h.to('cuda').float();cache=new_context(prefix)
    initial=torch.cat([metric.propose(h,64),rank(h,64)],dim=-1)
    tokens=[torch.tensor(128000,device='cuda')];ids_all=[];scores_all=[];rounds=[]
    for pos in range(1,len(h)):
        ids=initial[pos];bestscore=None;bestid=None;bestresponse=None;ir=[];sr=[]
        for step in range(3):
            predicted=candidates(prefix,cache,ids,pos).float();score=F.cosine_similarity(predicted,h[pos:pos+1],dim=-1)
            j=score.argmax()
            if bestscore is None or score[j]>bestscore:bestscore=score[j];bestid=ids[j];bestresponse=predicted[j]
            ir.append(ids);sr.append(score)
            query=(h[pos]-bestresponse+intrinsic[bestid].float()).unsqueeze(0)
            ids=rank(query,32)[0]
            if ids[0]==bestid:break
        tokens.append(bestid);prefix.run_cached(bestid.reshape(1,1),cache,pos)
        ids_all.append(F.pad(torch.cat(ir),(0,192-sum(len(v) for v in ir)),value=-1))
        scores_all.append(F.pad(torch.cat(sr),(0,192-sum(len(v) for v in sr)),value=-2))
        rounds.append(len(ir))
    out={'tokens':torch.stack(tokens).cpu(),'candidates':torch.stack(ids_all).cpu(),'scores':torch.stack(scores_all).cpu(),'rounds':torch.tensor(rounds)}
    sync();return out,time.perf_counter()-start
fixture=torch.tensor([[128000]+list(range(1000,1127))],device='cuda')
decode(prefix.forward_full(fixture)[0]);guard()
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
dst=OUT/'closure_dev_r1';dst.mkdir(exist_ok=False);entries=[]
for row in meta:
    guard();result,seconds=decode(obs[row['id']])
    path=dst/(row['id']+'.safetensors');save_file(result,str(path))
    entries.append({**row,'method':'closure','path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':seconds,'simulations':int((result['candidates']>=0).sum()),'round_counts':torch.bincount(result['rounds'],minlength=4).tolist()})
    print(row['id'],round(seconds,3),entries[-1]['round_counts'],flush=True)
write(X/'closure_dev_r1_freeze.json',{'environment':env,'entries':entries,'observations_sha256':digest(panel/'observations.safetensors'),'truth_read':False,'scope':'opened-panel development; not fresh','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
