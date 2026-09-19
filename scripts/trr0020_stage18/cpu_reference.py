"""Check full-vocabulary rank-one scores and the normalized-residual identity."""
import torch

def reference():
    torch.set_num_threads(2);torch.manual_seed(200044)
    dtype=torch.float64;p,d,v=4,7,29
    e=torch.randn(v,d,dtype=dtype);z=torch.randn(p,d,dtype=dtype);g=torch.randn_like(z)
    error=torch.tensor([.001,.03,.2,.8],dtype=dtype)
    checks=[]
    for metric in ['raw','white']:
        r=torch.eye(d,dtype=dtype) if metric=='raw' else torch.randn(d,d,dtype=dtype)+4*torch.eye(d,dtype=dtype)
        ri=torch.linalg.inv(r);norm=(e@r).square().sum(-1)
        delta=e[None]-z[:,None]
        for beta in [.01,.03,.1]:
            lam=(beta*(g@ri.T).square().sum(-1)/(2*error)).clamp(1e-6,1e6)
            linear=g@e.T-(g*z).sum(-1,keepdim=True)
            distance=(norm[None]+(z@r).square().sum(-1,keepdim=True)-2*((z@r)@r.T)@e.T).clamp_min(0)
            for gamma in [0.,1.]:
                score=-linear-.5*lam[:,None]*distance-gamma*linear.square()/(4*error[:,None])
                b=lam[:,None,None]*(r@r.T)[None]+gamma*g[:,:,None]*g[:,None,:]/(2*error[:,None,None])
                dense=-(g[:,None]*delta).sum(-1)-.5*torch.einsum('pvd,pde,pve->pv',delta,b,delta)
                torch.testing.assert_close(score,dense,rtol=1e-11,atol=1e-10)
                assert torch.equal(score.argmax(-1),dense.argmax(-1))
                checks.append({'metric':metric,'beta':beta,'gamma':gamma,'max_absolute_error':float((score-dense).abs().max()),'all_vocab_argmax_equal':True})
    a=torch.randn(5,d,dtype=dtype);x=torch.randn(d,dtype=dtype)
    normalize=lambda y:(a@y)/(a@y).norm()
    u=normalize(x);target=torch.randn(5,dtype=dtype);target=target/target.norm()
    residual=u-target;loss=.5*residual.square().sum();j=torch.autograd.functional.jacobian(normalize,x)
    gradient=j.T@residual;unit=residual/residual.norm()
    exact=j.T@torch.outer(unit,unit)@j;rank=torch.outer(gradient,gradient)/(2*loss)
    torch.testing.assert_close(exact,rank,rtol=1e-12,atol=1e-12)
    ev=torch.linalg.eigvalsh(j.T@j-rank)
    assert float(ev.min())>-1e-12
    emb=torch.tensor([-2.,-1.,0.,1.,2.],dtype=dtype)
    recovered=[]
    for target in [-2.,2.]:
        gradient=-target;loss=.5*target**2;linear=gradient*emb
        lam=.1*gradient**2/(2*loss)
        scores=-linear-linear.square()/(4*loss)-.5*lam*emb.square()
        recovered.append(int(scores.argmax()))
    assert recovered==[0,4]
    return {'passed':True,'seed':200044,'score_checks':checks,'residual_rank_identity_max_error':float((exact-rank).abs().max()),'remaining_Gauss_Newton_min_eigenvalue':float(ev.min()),'linear_edge_tokens_recovered':recovered,'scope':'tiny algebra and no vocabulary exclusion; not transformer efficacy'}
