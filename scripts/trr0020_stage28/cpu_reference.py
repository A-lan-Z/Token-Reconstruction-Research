"""Independent small dense metric and direction references."""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage27"))
from metric_direction import build,direction,RULES
from cheap_update import direction as old_direction
def reference():
    torch.set_num_threads(2);torch.manual_seed(200068);checks=[];metrics=[]
    for dtype in [torch.float64,torch.float32]:
        transform=torch.randn(8,8,dtype=dtype);transform[:,-1]=0
        ps,raw=build(transform)
        rr=transform.double()@transform.double().T
        expected=rr+1e-3*rr.trace()/8*torch.eye(8,dtype=torch.float64)
        inverse=torch.linalg.solve(expected,torch.eye(8,dtype=torch.float64))
        for key,ref in [("metric",expected),("inverse_metric",inverse)]:
            ref=ref/ref.diagonal().median();actual=ps[key].double()
            tol=2e-4 if dtype==torch.float32 else 2e-11
            torch.testing.assert_close(actual,ref,rtol=tol,atol=tol)
            values=torch.linalg.eigvalsh(actual)
            if not bool((values>0).all()):raise RuntimeError("nonpositive metric")
            metrics.append({"dtype":str(dtype),"metric":key,"max_abs_error":float((actual-ref).abs().max()),
              "min_eigenvalue":float(values.min()),"max_eigenvalue":float(values.max()),"passed":True})
        j=torch.randn(4,10,8,dtype=dtype);j[3]=0
        rhs=torch.randn(4,10,dtype=dtype);rhs[0]=0
        mv=lambda v:(j@v[...,None])[...,0]
        rmv=lambda v:(j.transpose(-1,-2)@v[...,None])[...,0]
        for metric,p in ps.items():
            for rule in RULES:
                actual,work=direction(rule,rhs,mv,rmv,p)
                if p is None:
                    expected,old_work=old_direction(rule,rhs,mv,rmv)
                    if not torch.equal(actual,expected):raise RuntimeError("identity output changed")
                    exact=True
                else:
                    # Independently evaluate each dense normal-equation line.
                    expected=torch.zeros_like(actual);exact=False
                    for row in [1,2]:
                        g=j[row].T@rhs[row];search=p@g;energy=g@search
                        if rule=="polyak0.25":alpha=.25*(rhs[row]@rhs[row])/energy
                        else:
                            ridge=.0001*(g@g)/(rhs[row]@rhs[row])
                            normal=j[row].T@j[row]+ridge*torch.eye(8,dtype=dtype)
                            alpha=energy/(search@normal@search)
                        expected[row]=alpha*search
                    tol=3e-5 if dtype==torch.float32 else 2e-12
                    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
                if not torch.equal(actual[[0,3]],torch.zeros_like(actual[[0,3]])):raise RuntimeError("zero gradient not preserved")
                checks.append({"dtype":str(dtype),"metric":metric,"rule":rule,"max_abs_error":float((actual-expected).abs().max()),
                  "identity_exact":exact,"zero_rhs_and_gradient_preserved":True,**work})
    for bad in [torch.zeros(8,8),torch.full((8,8),float("nan"))]:
        try:build(bad)
        except ValueError:continue
        raise RuntimeError("invalid metric accepted")
    return {"passed":True,"seed":200068,"metrics":metrics,"directions":checks,
      "invalid_metrics_rejected":2,"scope":"small dense algebra only; no reconstruction evidence"}
if __name__=="__main__":print(json.dumps(reference()))
