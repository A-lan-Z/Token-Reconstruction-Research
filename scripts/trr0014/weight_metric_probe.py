"""Weights-only structural proposal metric. Prediction code never opens labels."""
from native import *
from torch.nn import functional as F
import resource
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
env=environment();prefix=load_prefix();guard()
dst=X/'weight_metric_dev_r1';dst.mkdir(exist_ok=False)
obsfile=RESCUE/'development/observations.safetensors';obs=load_file(str(obsfile));meta=json.loads((RESCUE/'development/metadata.json').read_text())
dim=prefix.embed_tokens.weight.shape[-1]
@torch.no_grad()
def gram(kind):
    C=torch.zeros((dim,dim),device='cuda')
    for layer in prefix.layers:
        modules=[layer.mlp.down_proj] if kind=='mlp' else [layer.mlp.down_proj,layer.self_attn.o_proj]
        for module in modules:
            w=module.weight.float();c=w@w.T;C+=c/c.trace();del w,c
    return C
@torch.no_grad()
def decode(h,table,transform,method):
    sync();start=time.perf_counter()
    query=F.normalize(h.to('cuda').float()@transform,dim=-1)
    ids=(query@table.T).topk(64,dim=-1).indices
    cache=new_context(prefix);tokens=[128000];trace=[]
    for pos in range(1,len(h)):
        output=candidates(prefix,cache,ids[pos],pos).float()
        loss=(output-h[pos].to('cuda').float()).square().mean(-1)
        j=int(loss.argmin());v=int(ids[pos,j]);tokens.append(v)
        trace.append({'position':pos,'token':v,'checks':[{'token':int(v),'mse':float(l)} for v,l in zip(ids[pos],loss)]})
        prefix.run_cached(torch.tensor([[v]],device='cuda'),cache,pos)
    sync();return {'tokens':tokens,'trace':trace,'seconds':time.perf_counter()-start,'method':method}
for kind in ['total','diagonal','mlp']:
    guard();sync();start=time.perf_counter();C=gram(kind)
    if kind=='diagonal':
        values=C.diag();transform=torch.diag(values.rsqrt());eigen_seconds=0.
    else:
        sync();t=time.perf_counter();values,U=torch.linalg.eigh(C);sync();eigen_seconds=time.perf_counter()-t
        transform=U*values.clamp_min(torch.finfo(torch.float32).eps*values.max()).rsqrt()
    table=torch.empty_like(prefix.embed_tokens.weight,dtype=torch.float32)
    for lo in range(0,len(table),4096):
        table[lo:lo+4096]=F.normalize(prefix.embed_tokens.weight[lo:lo+4096].float()@transform,dim=-1)
    sync();setup=time.perf_counter()-start;guard()
    write(dst/(kind+'_setup.json'),{'seconds':setup,'eigen_seconds':eigen_seconds,'min_eigenvalue':float(values.min()),'max_eigenvalue':float(values.max()),'matrix_condition':float(values.max()/values.min()),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'fit_steps':0})
    for row in meta:
        result=decode(obs[row['id']],table,transform,kind);result['environment']=env
        write(dst/(kind+'_'+row['id']+'.json'),result);print(kind,row['id'],result['seconds'],flush=True)
    del table,transform,C
write(dst/'freeze.json',{'environment':env,'observations_sha256':digest(obsfile),'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(dst.glob('*.json'))],'truth_read':False,'retrospective_development':True})
