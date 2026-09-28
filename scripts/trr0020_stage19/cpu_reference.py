import torch

def reference():
    torch.set_num_threads(2);torch.manual_seed(200046);dt=torch.float64
    e=torch.randn(29,7,dtype=dt);q=torch.randn(4,7,dtype=dt)
    r=torch.randn(7,7,dtype=dt)+4*torch.eye(7,dtype=dt)
    checks=[]
    for name,transform in [('raw',torch.eye(7,dtype=dt)),('white',r)]:
        norm=(e@transform).square().sum(-1);dot=((q@transform)@transform.T)@e.T
        l2=2*dot-norm[None];cos=dot/norm.sqrt()[None]
        dense=-((q@transform)[:,None]-(e@transform)[None]).square().sum(-1)
        normalized=(q@transform)@torch.nn.functional.normalize(e@transform,dim=-1).T
        torch.testing.assert_close(l2-(q@transform).square().sum(-1,keepdim=True),dense,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(cos,normalized,rtol=1e-12,atol=1e-12)
        assert torch.equal(l2.argmax(-1),dense.argmax(-1))
        checks.append({'metric':name,'max_l2_error':float((l2-(q@transform).square().sum(-1,keepdim=True)-dense).abs().max()),'max_cosine_error':float((cos-normalized).abs().max())})
    emb=torch.tensor([[-1.,1.],[3.,1.],[.6,1.]],dtype=dt);p=torch.tensor([.6,.4,0.],dtype=dt)
    mean=p@emb;largest=int(p.argmax());nearest=int(((emb-mean).square().sum(-1)).argmin())
    assert largest==0 and nearest==2
    return {'passed':True,'seed':200046,'checks':checks,'counterexample':{'largest_weight_token':largest,'nearest_mean_token':nearest,'mean':mean.tolist()},'scope':'readout algebra and a counterexample to argmax equivalence; no model efficacy claim'}
