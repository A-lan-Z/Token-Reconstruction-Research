from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage29"))
from mirror_step import update as original
from adaptive_step import update
def reference():
    torch.set_num_threads(2);torch.manual_seed(200070);checks=[]
    for dtype in [torch.float64,torch.float32]:
        logits=torch.randn(5,17,dtype=dtype);gradient=torch.randn_like(logits)
        error=torch.tensor([0.,.0005,.0125,.05,2.],dtype=dtype)
        expected_factor=torch.tensor([0.,.1,.5,1.,1.],dtype=dtype)
        for tau in [.01,.1,1.]:
            actual,info=update(logits,gradient,tau,error)
            control,ci=original(logits,gradient,tau)
            torch.testing.assert_close(info["error_factor"],expected_factor)
            torch.testing.assert_close(info["rate"],ci["rate"]*expected_factor)
            if not torch.equal(actual[3:],control[3:]):raise RuntimeError("unit scaling changed original rule")
            p=torch.softmax(logits,dim=-1)
            reference=p*torch.exp(-info["rate"][:,None]*gradient)
            reference=reference/reference.sum(-1,keepdim=True)
            q=torch.softmax(actual,dim=-1);tol=2e-6 if dtype==torch.float32 else 2e-13
            torch.testing.assert_close(q,reference,rtol=tol,atol=tol)
            torch.testing.assert_close(q[0],p[0],rtol=tol,atol=tol)
            if not torch.isfinite(actual).all() or not (info["update_span"]<=4.00001*expected_factor).all():raise RuntimeError("invalid step")
            checks.append({"dtype":str(dtype),"tau":tau,"probability_max_error":float((q-reference).abs().max()),
              "unit_scaling_exact":True,"zero_error_preserved":True,"span_bound_passed":True})
        negative,_=update(logits,gradient,.1,-torch.ones(5,dtype=dtype))
        torch.testing.assert_close(torch.softmax(negative,dim=-1),torch.softmax(logits,dim=-1),rtol=tol,atol=tol)
    return {"passed":True,"seed":200070,"checks":checks,"negative_roundoff_error_checks":2,
      "scope":"tiny algebra and unit-scale equivalence; no prefix or reconstruction efficacy"}
if __name__=="__main__":print(json.dumps(reference()))
