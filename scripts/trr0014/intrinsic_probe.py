"""Candidate recall for context-free responses of the same prefix, no labels."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
env=environment();prefix=load_prefix();metric=PrefixWeightMetric(prefix);metric.build();guard()
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
dst=OUT/'intrinsic_dev_r1';dst.mkdir(exist_ok=False);entries=[];preps=[]
V,D=prefix.embed_tokens.weight.shape
# Largest dictionary batch already256-qualified. Both maps operate independently per token.
for kind in ['mlp_only','self_only']:
    sync();start=time.perf_counter()
    for lo in range(0,V,256):
        if lo%8192==0:guard()
        ids=torch.arange(lo,min(lo+256,V),device='cuda').unsqueeze(1)
        h=prefix.embed_tokens(ids)
        if kind=='self_only':
            h=prefix.forward_full(ids)
        else:
            for layer in prefix.layers:h=h+layer.mlp(layer.post_attention_layernorm(h))
        metric.table[lo:lo+len(ids)]=F.normalize(h[:,0].float()@metric.transform,dim=-1)
    sync();preps.append({'kind':kind,'seconds':time.perf_counter()-start})
    for row in meta:
        sync();start=time.perf_counter();ids=metric.propose(obs[row['id']].float(),256);tokens=ids[:,0].clone();tokens[0]=128000
        result={'tokens':tokens.cpu(),'candidates':ids[1:].cpu()};sync();seconds=time.perf_counter()-start
        path=dst/(row['id']+'__'+kind+'.safetensors');save_file(result,str(path))
        entries.append({**row,'method':kind,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':seconds})
    print(kind,'frozen',flush=True)
write(X/'intrinsic_dev_r1_freeze.json',{'environment':env,'entries':entries,'preparation':preps,'observations_sha256':digest(panel/'observations.safetensors'),'truth_read':False,'scope':'opened-panel proposal-recall only, K256 contains fixed K64 subset','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
