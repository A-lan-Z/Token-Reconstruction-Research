"""Whole-observation, forward-only sublayer reversal; no prediction labels."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
from torch.nn import functional as F
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
env=environment();prefix=load_prefix();metric=PrefixWeightMetric(prefix);metric.build()
emb=F.normalize(prefix.embed_tokens.weight.float(),dim=-1);guard()
# Known BOS states are legitimate and computed only from the supplied prefix.
bos=prefix.embed_tokens(torch.tensor([[128000]],device='cuda'));bst=[]
for layer in prefix.layers:
    pin=bos.clone();pe=prefix.rotary_emb(bos,torch.zeros((1,1),device='cuda',dtype=torch.long))
    mid=pin+layer.self_attn(layer.input_layernorm(pin),position_embeddings=pe)[0]
    bos=mid+layer.mlp(layer.post_attention_layernorm(mid))
    bst.append((pin,mid,bos.clone()))
@torch.no_grad()
def inverse_part(y,fn,known,kind):
    x=y.float().clone();x[:,0]=known.float()
    best=x.clone();bestloss=torch.full(y.shape[:2],float('inf'),device='cuda')
    xs=[];fs=[];trace=[]
    steps=6 if kind=='damped6' else 10
    for it in range(steps):
        fx=y.float()-fn(x.to(torch.bfloat16)).float()
        fx[:,0]=known.float()
        loss=(x-fx).square().mean(-1)
        good=loss<bestloss;best=torch.where(good.unsqueeze(-1),x,best);bestloss=torch.minimum(bestloss,loss)
        trace.append(float(loss[:,1:].mean()))
        if kind=='damped6':
            x=.5*x+.5*fx
        else:
            xs.append(x);fs.append(fx);xs=xs[-4:];fs=fs[-4:]
            if len(xs)<2:x=fx
            else:
                XX=torch.stack(xs,dim=2);FF=torch.stack(fs,dim=2);R=FF-XX
                G=R@R.transpose(-1,-2)
                scale=G.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
                G=G/scale.unsqueeze(-1).unsqueeze(-1)
                G=G+1e-4*torch.eye(len(xs),device='cuda')
                a=torch.linalg.solve(G,torch.ones((*G.shape[:-1],1),device='cuda'))
                a=a/a.sum(-2,keepdim=True).clamp_min(1e-12)
                x=(a*FF).sum(2)
        if not torch.isfinite(x).all():raise RuntimeError('nonfinite inverse attempt')
        x[:,0]=known.float()
    return best.to(torch.bfloat16),trace
@torch.no_grad()
def reverse(h,kind):
    y=h.to('cuda').unsqueeze(0).to(torch.bfloat16)
    pos=torch.arange(y.shape[1],device='cuda').reshape(1,-1)
    pe=prefix.rotary_emb(y,pos);mask=prefix._causal_mask(y,start_pos=0,total_tokens=y.shape[1]);tr=[]
    for i in reversed(range(len(prefix.layers))):
        layer=prefix.layers[i]
        y,t=inverse_part(y,lambda x:layer.mlp(layer.post_attention_layernorm(x)),bst[i][1],kind);tr.append({'layer':i,'part':'mlp','residual_mse':t})
        y,t=inverse_part(y,lambda x:layer.self_attn(layer.input_layernorm(x),position_embeddings=pe,attention_mask=mask)[0],bst[i][0],kind);tr.append({'layer':i,'part':'attention','residual_mse':t})
    return y[0].float(),tr
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
dst=OUT/'reverse_dev_r1';dst.mkdir(exist_ok=False);entries=[]
# Largest existing representative geometry128; no new candidate batch shape.
fixture=torch.tensor([[128000]+list(range(1000,1127))],device='cuda')
qual=[]
for kind in ['damped6','anderson10']:
    sync();start=time.perf_counter();q,tr=reverse(prefix.forward_full(fixture)[0],kind);sync()
    qual.append({'kind':kind,'seconds':time.perf_counter()-start,'finite':bool(torch.isfinite(q).all()),'trace':tr});guard()
for row in meta:
    for kind in ['damped6','anderson10']:
        guard();sync();start=time.perf_counter();q,tr=reverse(obs[row['id']],kind)
        for geometry in ['raw','metric']:
            ids=metric.propose(q,64) if geometry=='metric' else (F.normalize(q,dim=-1)@emb.T).topk(64,dim=-1).indices
            tokens=ids[:,0].clone();tokens[0]=128000
            result={'tokens':tokens.cpu(),'candidates':ids[1:].cpu()}
            sync();seconds=time.perf_counter()-start;method=kind+'_'+geometry
            path=dst/(row['id']+'__'+method+'.safetensors');save_file(result,str(path))
            entries.append({**row,'method':method,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':seconds,'trace':tr})
    print(row['id'],'frozen',flush=True)
write(X/'reverse_dev_r1_freeze.json',{'environment':env,'entries':entries,'qualification':qual,'observations_sha256':digest(panel/'observations.safetensors'),'truth_read':False,'scope':'opened-panel proposal-recall probe; top1 output only, not A2 reconstruction','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
