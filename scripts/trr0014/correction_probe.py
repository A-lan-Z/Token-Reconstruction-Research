"""Opened-panel forward-only residual subtraction probe; no truth in prediction."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
assert (X/'fresh_score.json').exists()
env=environment();prefix=load_prefix();metric=PrefixWeightMetric(prefix);metric.build();guard()
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
dst=OUT/'correction_dev_r1';dst.mkdir(exist_ok=False)
entries=[]
@torch.no_grad()
def decode(h,mode):
    sync();start=time.perf_counter();h=h.to('cuda').float()
    cache=new_context(prefix);tokens=[torch.tensor(128000,device='cuda')];proposals=[];score_rows=[];mse_rows=[]
    initial=metric.propose(h,16)
    for pos in range(1,len(h)):
        query=h[pos:pos+1];bestscore=None;bestid=None;bestresponse=None
        idrows=[];scores=[];mses=[]
        for step in range(3):
            ids=initial[pos] if step==0 else metric.propose(query,16)[0]
            predicted=candidates(prefix,cache,ids,pos).float()
            score=F.cosine_similarity(predicted,h[pos:pos+1],dim=-1)
            mse=(predicted-h[pos]).square().mean(-1)
            j=score.argmax()
            if bestscore is None or score[j]>bestscore:
                bestscore=score[j];bestid=ids[j];bestresponse=predicted[j]
            query=(h[pos]-bestresponse+prefix.embed_tokens.weight[bestid].float()).unsqueeze(0)
            idrows.append(ids);scores.append(score);mses.append(mse)
        tokens.append(bestid);prefix.run_cached(bestid.reshape(1,1),cache,pos)
        proposals.append(torch.cat(idrows));score_rows.append(torch.cat(scores));mse_rows.append(torch.cat(mses))
    out={'tokens':torch.stack(tokens).cpu(),'candidates':torch.stack(proposals).cpu(),'scores':torch.stack(score_rows).cpu(),'mse':torch.stack(mse_rows).cpu()}
    sync();return out,time.perf_counter()-start
# All 48 records remain, with unchanged conditions and no answer-dependent selection.
for row in meta:
    guard();result,seconds=decode(obs[row['id']],'correction3x16')
    path=dst/(row['id']+'.safetensors');save_file(result,str(path))
    entries.append({**row,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':seconds})
    print(row['id'],round(seconds,3),flush=True)
write(X/'correction_dev_r1_freeze.json',{'environment':env,'entries':entries,'observations_sha256':digest(panel/'observations.safetensors'),'method':'weight metric + three16candidate rounds with raw residual subtraction; best actually verified cosine retained','truth_read':False,'scope':'opened-panel development; original results unchanged','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
