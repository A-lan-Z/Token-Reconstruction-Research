"""Independent dense-autograd adjoint and dense least-squares references."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"trr0020_stage21"))
import torch
from linearized_prefix import forward,diagonal_jvp
from adjoint import diagonal_vjp
from cgls import least_squares
def reference():
    torch.set_num_threads(2);torch.manual_seed(200054);dt=torch.float64
    length,width,heads,kv,dim,mid=4,8,2,1,4,12
    layers=[]
    for _ in range(2):
        p={key:.12*torch.randn(out,inp,dtype=dt) for key,out,inp in [("q",8,8),("k",4,8),("v",4,8),("o",8,8),("gate",12,8),("up",12,8),("down",8,12)]}
        p.update(n1=1+.1*torch.randn(width,dtype=dt),n2=1+.1*torch.randn(width,dtype=dt),eps1=1e-5,eps2=1e-5,heads=heads,kv_heads=kv,head_dim=dim,groups=2,scale=dim**-.5)
        layers.append(p)
    phase=torch.randn(length,dim//2,dtype=dt);phase=torch.cat([phase,phase],-1);cos=phase.cos();sin=phase.sin()
    x=torch.randn(length,width,dtype=dt);_,cache=forward(x,layers,cos,sin)
    jac=torch.autograd.functional.jacobian(lambda v:forward(v,layers,cos,sin)[0],x)
    blocks=torch.stack([jac[i,:,i,:] for i in range(length)])
    errors=[];duality=[]
    for _ in range(5):
        left=torch.randn_like(x);right=torch.randn_like(x)
        actual=diagonal_vjp(left,layers,cache,cos,sin)
        exact=(blocks.transpose(-1,-2)@left[...,None])[...,0]
        torch.testing.assert_close(actual,exact,rtol=1e-11,atol=1e-11)
        jvp=diagonal_jvp(right,layers,cache,cos,sin)
        one=(left*jvp).sum(-1);two=(actual*right).sum(-1)
        torch.testing.assert_close(one,two,rtol=1e-11,atol=1e-11)
        errors.append(float((actual-exact).abs().max()));duality.append(float((one-two).abs().max()))
    b=.1*torch.randn_like(x);b[0]=0;linear=[]
    mv=lambda v:diagonal_jvp(v,layers,cache,cos,sin)
    rmv=lambda v:diagonal_vjp(v,layers,cache,cos,sin)
    for damp in [0.,1e-4]:
        value,stats=least_squares(mv,rmv,b,steps=16,damping=damp)
        matrix=blocks.transpose(-1,-2)@blocks+stats["ridge"][:,None,None]*torch.eye(width,dtype=dt)
        exact=torch.linalg.solve(matrix,(blocks.transpose(-1,-2)@b[...,None]))[...,0]
        torch.testing.assert_close(value,exact,rtol=1e-8,atol=1e-10)
        assert torch.equal(value[0],torch.zeros_like(value[0]))
        actual_residual=(mv(value)-b).norm(dim=-1)
        torch.testing.assert_close(stats["recurrence_residual_norm"],actual_residual,rtol=1e-5,atol=1e-12)
        linear.append({"damping":damp,"dense_direction_max_error":float((value-exact).abs().max()),"residual_norms":actual_residual.tolist()})
    return {"passed":True,"seed":200054,"dtype":"float64","vjp_max_errors":errors,"duality_max_errors":duality,"linear_checks":linear,"zero_rhs_preserved":True,"scope":"tiny synthetic algebra only; real-prefix adjoint and reconstruction unqualified"}
if __name__=="__main__":
    import json
    print(json.dumps(reference()))
