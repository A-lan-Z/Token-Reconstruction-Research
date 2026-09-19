"""Weights/forward-derived nuisance metrics; no fitted decoder or examples."""
from native import *
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
env=environment();prefix=load_prefix();V,D=prefix.embed_tokens.weight.shape
intrinsic=torch.empty_like(prefix.embed_tokens.weight)
for lo in range(0,V,256):
    if lo%8192==0:guard()
    h=prefix.embed_tokens.weight[lo:lo+256].unsqueeze(1)
    for layer in prefix.layers:h=h+layer.mlp(layer.post_attention_layernorm(h))
    intrinsic[lo:lo+len(h)]=h[:,0]
table=torch.empty((V,D),device='cuda',dtype=torch.float32)
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
dst=OUT/'nuisance_dev_r1';dst.mkdir(exist_ok=False);entries=[];stats=[]
for metric_kind in ['attention','unnormalized','response_delta']:
 for dictionary in ['embedding','mlp']:
    sync();start=time.perf_counter();C=torch.zeros((D,D),device='cuda');mean=torch.zeros(D,device='cuda')
    base=prefix.embed_tokens.weight if dictionary=='embedding' else intrinsic
    if metric_kind=='response_delta':
        response=load_file(str(OUT/'lookup_dev_r1_responses.safetensors'),device='cuda')['response']
        for lo in range(0,V,4096):
            r=response[lo:lo+4096]-base[lo:lo+4096].float()
            C+=r.T@r/V;mean+=r.sum(0)/V
        C-=mean[:,None]*mean[None,:];del response
    else:
        for layer in prefix.layers:
            modules=[layer.self_attn.o_proj] if metric_kind=='attention' else [layer.self_attn.o_proj,layer.mlp.down_proj]
            for module in modules:
                W=module.weight.float();G=W@W.T;C+=G/G.trace() if metric_kind=='attention' else G
    values,U=torch.linalg.eigh(C);A=U*values.clamp_min(torch.finfo(torch.float32).eps*values.max()).rsqrt()
    for lo in range(0,V,4096):table[lo:lo+4096]=F.normalize(base[lo:lo+4096].float()@A,dim=-1)
    sync();stats.append({'metric':metric_kind,'dictionary':dictionary,'min_eigenvalue':float(values.min()),'max_eigenvalue':float(values.max()),'build_seconds':time.perf_counter()-start})
    guard()
    for row in meta:
        q=F.normalize((obs[row['id']].to('cuda').float()-mean)@A,dim=-1)
        ids=(q@table.T).topk(256,dim=-1).indices;tokens=ids[:,0].clone();tokens[0]=128000
        method=metric_kind+'_'+dictionary;path=dst/(row['id']+'__'+method+'.safetensors')
        save_file({'tokens':tokens.cpu(),'candidates':ids[1:].cpu()},str(path))
        entries.append({**row,'method':method,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':0.0})
    print(metric_kind,dictionary,'frozen',flush=True)
write(X/'nuisance_dev_r1_freeze.json',{'environment':env,'entries':entries,'metric_stats':stats,'truth_read':False,'scope':'opened-panel proposal recall; no reconstruction/timing claim','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
