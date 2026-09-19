"""Dense-autograd reference for the normalized response equation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"trr0020_stage23"))
import torch
from cgls import least_squares
from normalized import equation
def reference():
    torch.set_num_threads(2);torch.manual_seed(200056);dt=torch.float64
    a=torch.eye(8,dtype=dt)[None]+.12*torch.randn(4,8,8,dtype=dt)
    x=torch.randn(4,8,dtype=dt);target=torch.randn(4,8,dtype=dt)
    mv=lambda v:(a@v[...,None])[...,0]
    rmv=lambda v:(a.transpose(-1,-2)@v[...,None])[...,0]
    prediction=mv(x);rhs,jvp,vjp=equation(prediction,target,mv,rmv)
    full=torch.autograd.functional.jacobian(lambda v:torch.nn.functional.normalize(mv(v),dim=-1),x)
    blocks=torch.stack([full[i,:,i,:] for i in range(4)])
    errors=[]
    for _ in range(5):
        v=torch.randn_like(x);w=torch.randn_like(x)
        exact_j=(blocks@v[...,None])[...,0];exact_t=(blocks.transpose(-1,-2)@w[...,None])[...,0]
        torch.testing.assert_close(jvp(v),exact_j,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(vjp(w),exact_t,rtol=1e-11,atol=1e-11)
        errors.append({"jvp_max":float((jvp(v)-exact_j).abs().max()),"vjp_max":float((vjp(w)-exact_t).abs().max())})
    rhs[0]=0;linear=[]
    for damping in [1e-4,1e-2]:
        value,stats=least_squares(jvp,vjp,rhs,steps=16,damping=damping)
        j=blocks[1:];ridge=stats["ridge"][1:]
        exact=torch.linalg.solve(j.transpose(-1,-2)@j+ridge[:,None,None]*torch.eye(8,dtype=dt),j.transpose(-1,-2)@rhs[1:,:,None])[...,0]
        torch.testing.assert_close(value[1:],exact,rtol=1e-7,atol=1e-8)
        assert torch.equal(value[0],torch.zeros_like(value[0]))
        linear.append({"damping":damping,"dense_direction_max_error":float((value[1:]-exact).abs().max())})
    return {"passed":True,"seed":200056,"dtype":"float64","derivative_checks":errors,"linear_checks":linear,"zero_rhs_preserved":True,"scope":"tiny normalized linear model only; real-prefix and reconstruction unqualified"}
if __name__=="__main__":
    import json
    print(json.dumps(reference()))
