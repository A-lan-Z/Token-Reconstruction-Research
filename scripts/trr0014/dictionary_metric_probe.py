"""Development only: combine prefix-generated dictionary and weight geometry."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
assert (X/'fresh_score.json').exists()
env=environment();prefix=load_prefix();metric=PrefixWeightMetric(prefix);metric.build();guard()
cache_receipt=json.loads((X/'lookup_dev_r1/cache.json').read_text())
cache_path=OUT/'lookup_dev_r1_responses.safetensors'
assert digest(cache_path)==cache_receipt['cache_sha256']
assert digest(ASSETS/'backup/prefix.safetensors')==cache_receipt['prefix_sha256']
response=load_file(str(cache_path),device='cuda')['response']
sync();start=time.perf_counter()
for lo in range(0,len(response),4096):
    metric.table[lo:lo+4096]=F.normalize(response[lo:lo+4096]@metric.transform,dim=-1)
sync();prepare_seconds=time.perf_counter()-start
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
dst=OUT/'dictionary_metric_dev_r1';dst.mkdir(exist_ok=False)
entries=[]
@torch.no_grad()
def decode(h,rounds,k):
    sync();start=time.perf_counter();h=h.to('cuda').float()
    cache=new_context(prefix);tokens=[torch.tensor(128000,device='cuda')];proposals=[];score_rows=[]
    initial=metric.propose(h,k)
    for pos in range(1,len(h)):
        query=h[pos:pos+1];bestscore=None;bestid=None;bestresponse=None
        idrows=[];scores=[]
        for step in range(rounds):
            ids=initial[pos] if step==0 else metric.propose(query,k)[0]
            predicted=candidates(prefix,cache,ids,pos).float()
            score=F.cosine_similarity(predicted,h[pos:pos+1],dim=-1)
            j=score.argmax()
            if bestscore is None or score[j]>bestscore:
                bestscore=score[j];bestid=ids[j];bestresponse=predicted[j]
            query=(h[pos]-bestresponse+response[bestid]).unsqueeze(0)
            idrows.append(ids);scores.append(score)
        tokens.append(bestid);prefix.run_cached(bestid.reshape(1,1),cache,pos)
        proposals.append(torch.cat(idrows));score_rows.append(torch.cat(scores))
    out={'tokens':torch.stack(tokens).cpu(),'candidates':torch.stack(proposals).cpu(),'scores':torch.stack(score_rows).cpu()}
    sync();return out,time.perf_counter()-start
for row in meta:
    for method,rounds,k in [('dictionary64',1,64),('transport3x16',3,16)]:
        guard();result,seconds=decode(obs[row['id']],rounds,k)
        path=dst/(row['id']+'__'+method+'.safetensors');save_file(result,str(path))
        entries.append({**row,'method':method,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':seconds})
    print(row['id'],'frozen',flush=True)
write(X/'dictionary_metric_dev_r1_freeze.json',{'environment':env,'entries':entries,'observations_sha256':digest(panel/'observations.safetensors'),'prepare_seconds':prepare_seconds,'truth_read':False,'scope':'opened-panel development; original results unchanged','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
