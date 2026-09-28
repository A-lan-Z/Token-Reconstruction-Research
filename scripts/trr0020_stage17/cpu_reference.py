"""CPU proof of cached full-vocabulary low-rank quadratic scoring."""
import json
import torch
torch.set_num_threads(2)

def reference():
    torch.manual_seed(200043)
    dtype=torch.float64;p,s,d,v=3,4,7,29
    probes=torch.randn(s,p,d,dtype=dtype)
    r=torch.randn(d,d,dtype=dtype)+4*torch.eye(d,dtype=dtype)
    rinv=torch.linalg.inv(r)
    e=torch.randn(v,d,dtype=dtype);z=torch.randn(p,d,dtype=dtype);g=torch.randn_like(z)
    lam=(probes@rinv.T).square().mean((0,2))
    results=[]
    for shrinkage in [.1,.5,1.]:
        sample=torch.einsum("spd,spe->pde",probes,probes)/s
        dense=(1-shrinkage)*sample+shrinkage*lam[:,None,None]*(r@r.T)[None]
        norms=torch.zeros(p,v,dtype=dtype)
        for probe in probes:norms+=(1-shrinkage)*(probe@e.T).square()/s
        norms+=shrinkage*lam[:,None]*(e@r).square().sum(-1)[None]
        for multiplier in [.1,1.]:
            bz=(1-shrinkage)*((probes*z[None]).sum(-1)[...,None]*probes).mean(0)
            bz+=shrinkage*lam[:,None]*(z@r@r.T)
            factored=(multiplier*bz-g)@e.T-.5*multiplier*norms
            delta=e[None]-z[:,None]
            direct=-(delta*g[:,None]).sum(-1)-.5*multiplier*torch.einsum("pvd,pde,pve->pv",delta,dense,delta)
            error=(factored-factored[:,:1]-direct+direct[:,:1]).abs().max()
            torch.testing.assert_close(factored-factored[:,:1],direct-direct[:,:1],rtol=1e-11,atol=1e-11)
            assert torch.equal(factored.argmax(-1),direct.argmax(-1))
            assert bool((torch.linalg.eigvalsh(dense)>0).all())
            results.append({"shrinkage":shrinkage,"multiplier":multiplier,"maximum_score_difference":float(error),
              "argmax_equal":True,"positive_definite":True})
    identity=torch.eye(d,dtype=dtype).unsqueeze(1).expand(d,p,d)*d**.5
    target=e[torch.tensor([0,7,v-1])]
    gradient=z-target
    norm=e.square().sum(-1)
    result=(z-gradient)@e.T-.5*norm
    assert torch.equal(result.argmax(-1),torch.tensor([0,7,v-1]))
    return {"seed":200043,"device":"cpu","dtype":"float64","vocabulary_entries":v,
      "results":results,"linear_identity_recovers_first_middle_last_token":True,
      "one_cached_quadratic_surface_covers_the_full_vocabulary":True,"passed":True}
if __name__=="__main__":print(json.dumps(reference(),indent=2))
