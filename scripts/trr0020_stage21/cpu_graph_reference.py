"""Dense and earlier-solver checks for the graph-compatible linear direction."""
import torch
from krylov_graph import direction
from linearized_prefix import krylov_least_squares
def reference():
    torch.set_num_threads(2);torch.manual_seed(200050)
    a=2*torch.eye(8,dtype=torch.float64)[None]+.15*torch.randn(4,8,8,dtype=torch.float64)
    b=torch.randn(4,8,dtype=torch.float64);b[0]=0
    mv=lambda v:(a@v[...,None])[...,0]
    got,projected,info=direction(mv,b,8,ridge=1e-10)
    old,stats=krylov_least_squares(mv,b,8,ridge=1e-10)
    exact=torch.linalg.solve(a,b[...,None])[...,0]
    assert torch.equal(got,old) and bool((info==0).all())
    torch.testing.assert_close(got,exact,rtol=1e-7,atol=1e-9)
    torch.testing.assert_close(projected,stats["residual_norm"],rtol=1e-3,atol=1e-12)
    zero,_,zero_info=direction(lambda v:v,torch.zeros_like(b),8)
    assert torch.equal(zero,torch.zeros_like(zero)) and bool((zero_info==0).all())
    return {"passed":True,"seed":200050,"dtype":"float64","previous_direction_identical":True,"dense_direction_max_error":float((got-exact).abs().max()),"projected_vs_true_residual_max_error":float((projected-stats["residual_norm"]).abs().max()),"zero_rhs_preserved":True,"scope":"CPU linear algebra only; GPU replay and reconstruction not qualified"}
if __name__=="__main__":
    import json
    print(json.dumps(reference()))
