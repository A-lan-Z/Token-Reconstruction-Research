"""Dense reference for reverse-order approximate inverse composition."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"trr0020_stage21"))
import torch
from factorized import solve_factors
def reference():
    torch.set_num_threads(2);torch.manual_seed(200052)
    factors=[1.5*torch.eye(8,dtype=torch.float64)[None]+.12*torch.randn(4,8,8,dtype=torch.float64) for _ in range(4)]
    b=torch.randn(4,8,dtype=torch.float64);b[0]=0
    products=[lambda value,a=a:(a@value[...,None])[...,0] for a in factors]
    actual,residuals,info=solve_factors(products,b,8,ridge=1e-10)
    full=factors[0]
    for factor in factors[1:]:full=factor@full
    exact=torch.linalg.solve(full,b[...,None])[...,0]
    torch.testing.assert_close(actual,exact,rtol=1e-7,atol=1e-9)
    assert bool((info==0).all()) and torch.equal(actual[0],torch.zeros_like(actual[0]))
    return {"passed":True,"seed":200052,"dtype":"float64","factors":4,"dimension":8,"positions":4,
      "dense_direction_max_error":float((actual-exact).abs().max()),"reverse_layer_order_verified":True,
      "zero_rhs_preserved":True,"scope":"tiny linear factors only; actual-prefix benefit untested"}
if __name__=="__main__":
    import json
    print(json.dumps(reference()))
