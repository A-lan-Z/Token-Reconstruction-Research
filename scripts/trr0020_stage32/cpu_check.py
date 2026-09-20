"""CPU controls for observed-error budgets and deliberately approximate path solves."""
from pathlib import Path
import sys,json,traceback,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage31"))
from kl_step import update as original
from cpu_reference import reference_row,direct
from budget_step import path_update,update
def reference():
 torch.set_num_threads(2);torch.manual_seed(200072);checks=[];failures=[];anchors=[]
 for dtype in [torch.float64,torch.float32]:
  z=torch.randn(6,37,dtype=dtype)*3;g=torch.randn_like(z)
  for tau in [.01,.1,1.]:
   a,ai=original(z,g,tau);b,bi=path_update(z,g,tau,16)
   eq=torch.equal(a,b) and all(torch.equal(v,bi[k]) for k,v in ai.items())
   anchors.append({"dtype":str(dtype),"tau":tau,"exact":eq})
   if not eq:failures.append({"kind":"scalar control","dtype":str(dtype),"tau":tau})
  error=torch.tensor([-1e-7,0.,1e-4,.01,.2,1.],dtype=dtype)
  for factor in [.5,1.,2.]:
   for iterations in [4,8,16]:
    try:
     new,info=update(z,g,error,factor,iterations);actual=direct(z,new);budget=info["requested_budget"].double()
     tol=2e-10 if dtype==torch.float64 else 2e-5
     if not torch.isfinite(new).all() or not all(torch.isfinite(v).all() for v in info.values()):raise AssertionError("nonfinite")
     if (actual>budget+tol).any() or (actual < -tol).any():raise AssertionError("budget exceeded")
     torch.testing.assert_close(new[:2].softmax(-1),z[:2].softmax(-1),atol=tol,rtol=tol)
     root_errors=[]
     for row in range(2,len(z)):
      step,_=reference_row(z[row],g[row],float(info["target"][row]))
      root_errors.append(abs(float(info["normalized_step"][row])-step)/step)
     if iterations==16 and max(root_errors)>.004:raise AssertionError("qualified16root changed")
     checks.append({"dtype":str(dtype),"factor":factor,"iterations":iterations,"direct_kl":actual.tolist(),
       "budget":budget.tolist(),"max_relative_root_error":max(root_errors),
       "budget_use":(actual[2:]/budget[2:]).tolist(),"zero_negative_error_unchanged":True})
    except Exception:failures.append({"dtype":str(dtype),"factor":factor,"iterations":iterations,"failure":traceback.format_exc()})
 return {"passed":not failures,"seed":200072,"scalar_control_anchors":anchors,"checks":checks,"failures":failures,
  "scope":"variable-budget feasibility and independent root diagnostics;4/8 iterations are distinct approximate rules"}
if __name__=="__main__":
 r=reference();print(json.dumps(r));raise SystemExit(0 if r["passed"] else 1)
