"""Independent FP64 probability/path checks; never reads reconstruction labels."""
import json,traceback,torch
from kl_step import update,ITERATIONS,BUDGET_MARGIN
def direct(logits,new):
    a=logits.double().log_softmax(-1);b=new.double().log_softmax(-1)
    return (a.exp()*(a-b)).sum(-1)
def reference_row(z,g,tau):
    z=z.double();g=g.double()
    span=g.max()-g.min()
    if span==0:return None
    d=(g-g.min())/span;p=z.softmax(-1);a=z.log_softmax(-1)
    slope=(p*d).sum()
    if slope<1e-280:return None
    hi=float((tau-a[g.argmin()])/slope)
    if not torch.isfinite(torch.tensor(hi,dtype=torch.float64)):return None
    lo=0.
    for _ in range(160):
        mid=(lo+hi)/2
        value=float((p*(a-(z-mid*d).log_softmax(-1))).sum())
        if value<=tau:lo=mid
        else:hi=mid
    return lo,(z-lo*d).softmax(-1)
def reference():
    torch.set_num_threads(2);torch.manual_seed(200071)
    fixtures=[]
    for size in [17,1025]:
        z=torch.randn(5,size,dtype=torch.float64)*3;g=torch.randn_like(z)
        fixtures.append((f"random{size}",z,g))
    z=torch.tensor([[0.,-80.,-120.,-1000.],[0.,-80.,-120.,-1000.],
        [1000.,999.,-1000.,0.],[0.,0.,0.,0.],[0.,-1.,-2.,-3.],
        [0.,-1.,-2.,-3.]],dtype=torch.float64)
    g=torch.tensor([[1.,1.,1.,0.],[0.,1.,2.,3.],[1.,.5,0.,.9],
        [2.,2.,2.,2.],[1.,2.,3.,4.],[1e-12,2e-12,3e-12,4e-12]],dtype=torch.float64)
    fixtures.append(("extreme",z,g))
    checks=[];failures=[]
    for dtype in [torch.float64,torch.float32]:
      for name,z,g in fixtures:
       for tau in [.01,.1,1.]:
        try:
          a,b=z.to(dtype),g.to(dtype);new,info=update(a,b,tau)
          actual=direct(a,new)
          if not torch.isfinite(new).all() or not all(torch.isfinite(v).all() for v in info.values()):raise AssertionError("nonfinite")
          tol=2e-10 if dtype==torch.float64 else 2e-5
          if (actual>tau+tol).any() or (actual < -tol).any():raise AssertionError("direct KL outside budget")
          if not (info["initial_upper_kl"][info["active"]] >= tau*(1-BUDGET_MARGIN)-tol).all():raise AssertionError("invalid upper bracket")
          rowerrors=[];usage=[]
          for row in range(len(a)):
            ref=reference_row(a[row],b[row],tau*(1-BUDGET_MARGIN))
            if ref is None:continue
            step,q=ref
            if not info["active"][row]:continue
            rel=abs(float(info["normalized_step"][row])-step)/max(step,1e-30)
            rowerrors.append(rel);usage.append(float(actual[row]/tau))
            if rel>.004:raise AssertionError(f"step differs from independent root {row}: {rel}")
            torch.testing.assert_close(new[row].double().softmax(-1),q,rtol=.01,atol=.002)
            rates=torch.tensor([0.,step*.25,step*.5,step,step*2],dtype=torch.float64)
            zz=a[row].double();gg=b[row].double();gg=(gg-gg.min())/(gg.max()-gg.min())
            values=direct(zz.expand(5,-1),zz[None]-rates[:,None]*gg)
            if not (values[1:]>=values[:-1]-1e-12).all():raise AssertionError("nonmonotonic path")
          checks.append({"dtype":str(dtype),"fixture":name,"tau":tau,"direct_kl":actual.tolist(),
            "formula_kl":info["formula_kl"].tolist(),"max_relative_root_error":max(rowerrors,default=0.),
            "min_active_budget_use":min(usage,default=0.),"active":info["active"].tolist()})
        except Exception:
          failures.append({"dtype":str(dtype),"fixture":name,"tau":tau,"failure":traceback.format_exc()})
    return {"passed":not failures,"seed":200071,"iterations":ITERATIONS,"checks":checks,"failures":failures,
      "scope":"independent scalar root, direct FP64 probability KL, finite bracket, monotonicity, extreme finite logits; no reconstruction"}
if __name__=="__main__":
    result=reference();print(json.dumps(result));raise SystemExit(0 if result["passed"] else 1)
