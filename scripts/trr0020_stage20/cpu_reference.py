import torch

def reference():
    torch.set_num_threads(2);torch.manual_seed(200047);dt=torch.float64
    a=torch.randn(4,6,dtype=dt);x=torch.randn(3,6,dtype=dt,requires_grad=True);target=torch.nn.functional.normalize(torch.randn(3,4,dtype=dt),dim=-1)
    prediction=torch.nn.functional.normalize(x@a.T,dim=-1);error=1-(prediction*target).sum(-1)
    checks=[]
    for power in [1,2,4]:
        loss=error.mean() if power==1 else error.pow(power).mean()/(power*.05**(power-1))
        actual=torch.autograd.grad(loss,x,retain_graph=True)[0]
        weights=torch.ones_like(error) if power==1 else (error/.05).pow(power-1)
        expected=torch.autograd.grad(error,x,grad_outputs=weights/len(error),retain_graph=True)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        checks.append({'power':power,'max_gradient_error':float((actual-expected).abs().max())})
    values=torch.tensor([.001,.05,.5],dtype=dt)
    return {'passed':True,'seed':200047,'gradient_checks':checks,'relative_error_gradient_weights':{str(p):(values/.05).pow(p-1).tolist() for p in [1,2,4]},'scope':'scalar loss and chain-rule checks only; no efficacy claim'}
