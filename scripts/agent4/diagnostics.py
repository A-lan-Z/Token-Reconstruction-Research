from common import *
from solver import *
from safetensors.torch import save_file
import gc

def main():
    torch.set_num_threads(2)
    pre=environment()
    free,total=torch.cuda.mem_get_info()
    assert free>=8*2**30
    rows=[]
    for name,dtype in [('float32',torch.float32),('bfloat16',torch.bfloat16)]:
        start=time.perf_counter();prefix=load_prefix(dtype);load_s=time.perf_counter()-start
        guard();torch.cuda.reset_peak_memory_stats()
        ids=torch.tensor([[128000,2028,374,264,1296,13,220,16]],device='cuda')
        with torch.no_grad():
            obs=prefix.forward_full(ids).clone()
            cont=continuous_forward(prefix,prefix.embed_tokens(ids)).detach()
            iterative=torch.stack([prefix.forward_full(ids[:,:i+1])[0,-1] for i in range(ids.shape[1])]).unsqueeze(0)
            cache=prefix.new_cache()
            cached=torch.cat([prefix.run_cached(ids[:,i:i+1],cache,i) for i in range(ids.shape[1])],1)
        z=prefix.embed_tokens(ids)[:,-1].float().detach().clone().requires_grad_()
        z.data.add_(.001)
        hidden=torch.cat([prefix.embed_tokens(ids)[:,:-1].detach(),z.to(dtype).unsqueeze(1)],1)
        prediction=continuous_forward(prefix,hidden)[0,-1].float()
        loss=(prediction-obs[0,-1].float()).square().mean()
        grad,=torch.autograd.grad(loss,z)
        assert torch.isfinite(grad).all() and grad.norm()>0
        assert torch.equal(obs,cont)
        # Qualify biggest proposed geometry even when small feasibility fails.
        largest=ids.repeat(1,16)
        z128=prefix.embed_tokens(largest).detach().clone().requires_grad_()
        p=continuous_forward(prefix,z128); p[:,-1].float().square().mean().backward()
        guard();sync(torch.device('cuda'))
        rows.append({'dtype':name,'load_seconds':load_s,'continuous_discrete_exact':torch.equal(obs,cont),
            'full_vs_iterative_mse':(obs.float()-iterative.float()).square().mean(-1).tolist(),
            'full_vs_cached_mse':(obs.float()-cached.float()).square().mean(-1).tolist(),
            'gradient_norm':float(grad.norm()),'perturbed_loss':float(loss.detach()),
            '128_position_backward_finite':bool(torch.isfinite(z128.grad).all()),
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        # Independent load from actual restored files; exact state and output check.
        del z128,p,hidden,z,prediction,loss,grad,cache,cont,iterative,cached
        reference=obs.cpu();del prefix,obs;gc.collect();torch.cuda.empty_cache()
        restored=load_prefix(dtype,asset_root=OUT/'restore')
        assert torch.equal(restored.forward_full(ids).cpu(),reference)
        rows[-1]['clean_restore_output_exact']=True
        del restored;gc.collect();torch.cuda.empty_cache()
    write(EVID/'numerical_diagnostics.json',{'environment':pre,'diagnostics':rows,'end_utc':environment()['utc']})
    print(json.dumps(rows))
if __name__=='__main__':main()
