from common import *
from solver import continuous_forward
import gc
torch.set_num_threads(2);rows=[]
for name in ['float32','bfloat16']:
    p=load_prefix(getattr(torch,name),device='cpu')
    ids=torch.tensor([[128000,2028,374,264]])
    obs=p.forward_full(ids).clone()
    cont=continuous_forward(p,p.embed_tokens(ids))
    assert torch.equal(obs,cont)
    iterative=torch.stack([p.forward_full(ids[:,:i+1])[0,-1] for i in range(len(ids[0]))])
    z=p.embed_tokens(ids).detach().clone();z[0,-1]+=.001;z.requires_grad_()
    loss=(continuous_forward(p,z)[0,-1].float()-obs[0,-1].float()).square().mean()
    grad,=torch.autograd.grad(loss,z)
    rows.append({'dtype':name,'continuous_discrete_exact':True,'full_iterative_mse':(obs[0].float()-iterative.float()).square().mean(-1).tolist(),'gradient_norm':float(grad[0,-1].float().norm()),'perturbed_mse':float(loss.detach())})
    del p,obs,cont,iterative,z,loss,grad;gc.collect()
write(EVID/'cpu_public_diagnostic.json',{'rows':rows,'environment':environment(),'cpu_threads':2})
print(json.dumps(rows))
