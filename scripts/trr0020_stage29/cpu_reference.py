"""Independent probability-space and constrained-objective checks."""
import json,torch
from mirror_step import update
def reference():
    torch.set_num_threads(2);torch.manual_seed(200069);checks=[]
    for dtype in [torch.float64,torch.float32]:
        logits=torch.randn(4,13,dtype=dtype)*2
        gradient=torch.randn_like(logits);gradient[0]=0;gradient[1]=3
        for scale in [.01,.1,1.]:
            result,stats=update(logits,gradient,scale)
            p=torch.softmax(logits,dim=-1);actual=torch.softmax(result,dim=-1)
            reference=p*torch.exp(-stats["rate"][:,None]*gradient)
            reference=reference/reference.sum(-1,keepdim=True)
            tol=2e-6 if dtype==torch.float32 else 3e-13
            torch.testing.assert_close(actual,reference,rtol=tol,atol=tol)
            if not torch.isfinite(result).all() or not (actual>0).all():raise RuntimeError("vocabulary eligibility lost")
            if not (stats["update_span"]<=4.000001).all():raise RuntimeError("span cap")
            torch.testing.assert_close(actual[:2],p[:2],rtol=tol,atol=tol)
            # KKT: G + (log(q)-log(p)+1)/eta is constant across vocabulary.
            q=actual[2:].double();prior=p[2:].double();g=gradient[2:].double();eta=stats["rate"][2:].double()
            stationarity=g+(q.log()-prior.log()+1)/eta[:,None]
            stationarity=stationarity-stationarity.mean(-1,keepdim=True)
            limit=1e-5 if dtype==torch.float32 else 3e-12
            if float(stationarity.abs().max())>limit:raise RuntimeError("mirror KKT reference")
            # Convex KL-regularized linear objective must be no worse than no move.
            objective=(q*g).sum(-1)+(q*(q.log()-prior.log())).sum(-1)/eta
            unchanged=(prior*g).sum(-1)
            if not (objective<=unchanged+limit).all():raise RuntimeError("proximal objective increased")
            checks.append({"dtype":str(dtype),"kl_scale":scale,"probability_reference_max_error":float((actual-reference).abs().max()),
              "max_kkt_residual":float(stationarity.abs().max()),"proximal_objective_improved":True,"all_entries_positive":True,
              "zero_and_constant_gradient_preserved":True})
        extreme=torch.tensor([[80.,-80.,0.]],dtype=dtype);g=torch.tensor([[1.,-1.,0.]],dtype=dtype)
        z,stats=update(extreme,g,.1)
        if not torch.isfinite(z).all() or not torch.equal(z.argmax(-1),torch.tensor([0])):raise RuntimeError("finite-logit underflow handling")
        again,_=update(z,g,.1)
        if not (again[0,1]-again[0,0]>z[0,1]-z[0,0]):raise RuntimeError("low-probability token cannot recover")
    return {"passed":True,"seed":200069,"checks":checks,"extreme_logit_checks":2,
      "scope":"small full-vocabulary mirror-step math; no prefix, GPU or reconstruction qualification"}
if __name__=="__main__":print(json.dumps(reference()))
