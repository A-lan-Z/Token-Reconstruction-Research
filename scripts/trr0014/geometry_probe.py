"""Test two untuned comparison geometries of deterministic prefix dictionaries."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
env=environment();prefix=load_prefix();metric=PrefixWeightMetric(prefix);metric.build();del metric.table;metric.table=None
V,D=prefix.embed_tokens.weight.shape;table=torch.empty((V,D),device='cuda',dtype=torch.float32)
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
dst=OUT/'geometry_dev_r1';dst.mkdir(exist_ok=False);entries=[]
for kind in ['embedding','mlp','bos']:
    raw=load_file(str(OUT/'lookup_dev_r1_responses.safetensors'),device='cuda')['response'] if kind=='bos' else None
    for lo in range(0,V,256):
        if lo%8192==0:guard()
        h=prefix.embed_tokens.weight[lo:lo+256].unsqueeze(1)
        if kind=='mlp':
            for layer in prefix.layers:h=h+layer.mlp(layer.post_attention_layernorm(h))
        if kind=='bos':h=raw[lo:lo+256].unsqueeze(1)
        table[lo:lo+len(h)]=h[:,0].float()@metric.transform
    if raw is not None:del raw
    norms=table.square().sum(-1);mean=table.mean(0)
    for row in meta:
        q=obs[row['id']].to('cuda').float()@metric.transform
        ids=(2*q@table.T-norms).topk(256,dim=-1).indices
        tokens=ids[:,0].clone();tokens[0]=128000
        method=kind+'_euclidean';path=dst/(row['id']+'__'+method+'.safetensors')
        save_file({'tokens':tokens.cpu(),'candidates':ids[1:].cpu()},str(path))
        entries.append({**row,'method':method,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':0.0})
    table-=mean;table=F.normalize(table,dim=-1)
    for row in meta:
        q=F.normalize(obs[row['id']].to('cuda').float()@metric.transform-mean,dim=-1)
        ids=(q@table.T).topk(256,dim=-1).indices
        tokens=ids[:,0].clone();tokens[0]=128000
        method=kind+'_centered';path=dst/(row['id']+'__'+method+'.safetensors')
        save_file({'tokens':tokens.cpu(),'candidates':ids[1:].cpu()},str(path))
        entries.append({**row,'method':method,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':0.0})
    print(kind,'frozen',flush=True)
write(X/'geometry_dev_r1_freeze.json',{'environment':env,'entries':entries,'truth_read':False,'scope':'opened-panel proposal recall only; no runtime claim','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
