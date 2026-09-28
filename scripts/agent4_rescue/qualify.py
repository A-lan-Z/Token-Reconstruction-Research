"""Real Llama current-token values/gradients, batch and derivative preflight."""
from shared import *
from executor import CurrentToken
from solver import continuous_forward
from torch.nn.attention import sdpa_kernel,SDPBackend
import gc,traceback
torch.set_num_threads(2)
rows=[];env=environment()
for dtype in [torch.float32,torch.bfloat16]:
    prefix=load_prefix(dtype);guard()
    ids=([128000,2028,374,264,1296,13,220,16]*16)+[13]
    ex=CurrentToken(prefix)
    for pos in range(1,128):
        if pos in [1,4,15,39,127]:
            z=prefix.embed_tokens.weight[ids[pos]].float().detach().clone().requires_grad_()
            target=torch.zeros_like(z)
            allhidden=torch.cat((prefix.embed_tokens(torch.tensor([ids[:pos]],device='cuda')).detach(),z.to(dtype).view(1,1,-1)),1)
            y=continuous_forward(prefix,allhidden)[0,-1].float()
            g,=torch.autograd.grad((y-target).square().mean(),z)
            before={i:(k.clone(),v.clone()) for i,(k,v) in ex.state.items()}
            z2=z.detach().clone().requires_grad_();y2=ex(z2).float();g2,=torch.autograd.grad(y2.square().mean(),z2)
            rel=lambda a,b:float((a-b).norm()/a.norm().clamp_min(1e-12))
            unchanged=all(torch.equal(k,before[i][0]) and torch.equal(v,before[i][1]) for i,(k,v) in ex.state.items())
            assert unchanged and torch.isfinite(g2).all()
            row={'dtype':str(dtype),'position':pos,'forward_equal':torch.equal(y,y2),'forward_relative_error':rel(y,y2),'forward_mse':float((y-y2).square().mean()),'gradient_relative_error':rel(g,g2),'gradient_cosine':float(torch.nn.functional.cosine_similarity(g,g2,dim=0)),'cache_unchanged':unchanged}
            # Batch substitution changes GEMM geometry; only exact agreement permits use.
            zz=prefix.embed_tokens.weight[torch.tensor(ids[1:9],device='cuda')].float()
            with torch.no_grad():
                bat=ex(zz);single=torch.stack([ex(v) for v in zz])
            row['batch8_exact']=torch.equal(bat,single);row['batch8_mse']=float((bat-single).float().square().mean())
            rows.append(row)
        if pos<127:ex.commit(ids[pos])
    guard()
    if dtype==torch.bfloat16:
        inverse={}
        try:
            z=prefix.embed_tokens.weight[13].float().detach()
            v=torch.ones_like(z)/z.numel()**.5
            with sdpa_kernel(SDPBackend.MATH):
                t=time.perf_counter()
                y,jv=torch.func.jvp(lambda q:ex(q).float()/q.numel()**.5,(z,),(v,))
                yy,vjp=torch.func.vjp(lambda q:ex(q).float()/q.numel()**.5,z)
                jt=vjp(jv)[0];sync()
                with torch.no_grad():normal=ex(z).float()/z.numel()**.5
            assert torch.isfinite(jt).all()
            inverse={'passed':True,'jvp_vjp_seconds':time.perf_counter()-t,'math_forward_relative_error':rel(normal,yy),'jvp_norm':float(jv.norm()),'jtj_norm':float(jt.norm())}
        except Exception as error:
            inverse={'passed':False,'error':repr(error),'traceback':traceback.format_exc()}
    del ex,prefix;gc.collect();torch.cuda.empty_cache()
write(E/'executor_qualification.json',{'environment':env,'rows':rows,'inverse':inverse,'peak_reserved':torch.cuda.max_memory_reserved(),'batching_allowed':all(r['batch8_exact'] for r in rows if r['dtype']=='torch.bfloat16'),'classification':'Numerical port unless all values/gradients and subsequent search decisions agree; original BF16 observations retained.'})
print(json.dumps({'rows':rows,'inverse':inverse},indent=2))
