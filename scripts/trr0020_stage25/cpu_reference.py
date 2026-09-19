"""Independent dense-autograd checks of causal products and global CGLS."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
for path in ["scripts/trr0020_stage21","scripts/trr0020_stage23","scripts/trr0020_stage24"]:
    sys.path.insert(0,str(ROOT/path))
import torch
from torch.nn import functional as F
from linearized_prefix import forward
from causal import operators,free_positions
from global_cgls import least_squares
from normalized import equation

def reference():
    torch.set_num_threads(2);torch.manual_seed(200060);dt=torch.float64
    length,width,dim=4,8,4
    layers=[]
    for _ in range(2):
        p={key:.12*torch.randn(out,inp,dtype=dt) for key,out,inp in
          [("q",8,8),("k",4,8),("v",4,8),("o",8,8),("gate",12,8),("up",12,8),("down",8,12)]}
        p.update(n1=1+.1*torch.randn(width,dtype=dt),n2=1+.1*torch.randn(width,dtype=dt),
          eps1=1e-5,eps2=1e-5,heads=2,kv_heads=1,head_dim=dim,groups=2,scale=dim**-.5)
        layers.append(p)
    phase=torch.randn(length,dim//2,dtype=dt);phase=torch.cat([phase,phase],-1)
    cos,sin=phase.cos(),phase.sin();x=torch.randn(length,width,dtype=dt)
    y,cache=forward(x,layers,cos,sin);raw_mv,raw_rmv=operators(layers,cache,cos,sin)
    rawjac=torch.autograd.functional.jacobian(lambda v:forward(v,layers,cos,sin)[0],x)
    if not all(torch.count_nonzero(rawjac[i,:,j,:])==0 for i in range(length) for j in range(i+1,length)):
        raise RuntimeError("noncausal reference")
    coupling=float(rawjac[2,:,1,:].norm());assert coupling>0
    target=y+.1*torch.randn_like(y)
    checks=[];linear=[]
    for mode in ["raw","normalized"]:
        if mode=="raw":
            jac=rawjac.reshape(length*width,length*width).clone()
            mv,rmv=raw_mv,raw_rmv;b=free_positions(target-y)
        else:
            jac=torch.autograd.functional.jacobian(lambda v:F.normalize(forward(v,layers,cos,sin)[0],dim=-1),x).reshape(length*width,length*width)
            b,mv,rmv=equation(y,target,raw_mv,raw_rmv);b=free_positions(b)
        jac[:,:width]=0
        for rep in range(5):
            left=torch.randn_like(x);right=torch.randn_like(x)
            j=mv(right);jt=rmv(left)
            expected_j=(jac@right.flatten()).reshape_as(x)
            expected_jt=(jac.T@left.flatten()).reshape_as(x)
            torch.testing.assert_close(j,expected_j,rtol=1e-11,atol=1e-11)
            torch.testing.assert_close(jt,expected_jt,rtol=1e-11,atol=1e-11)
            torch.testing.assert_close((left*j).sum(),(jt*right).sum(),rtol=1e-11,atol=1e-11)
            checks.append({"mode":mode,"rep":rep,"jvp_max_error":float((j-expected_j).abs().max()),"vjp_max_error":float((jt-expected_jt).abs().max())})
        for damping in [.0001,.01]:
            value,stats=least_squares(mv,rmv,b,64,damping)
            exact=torch.linalg.solve(jac.T@jac+stats["ridge"]*torch.eye(length*width,dtype=dt),jac.T@b.flatten()).reshape_as(x)
            torch.testing.assert_close(value,exact,rtol=1e-7,atol=1e-9)
            true=(mv(value)-b).norm(dim=-1)
            torch.testing.assert_close(stats["recurrence_residual_norm"],true,rtol=1e-6,atol=1e-11)
            assert torch.equal(value[0],torch.zeros_like(value[0]))
            zero,_=least_squares(mv,rmv,torch.zeros_like(b),8,damping)
            assert torch.equal(zero,torch.zeros_like(zero))
            linear.append({"mode":mode,"damping":damping,"dense_direction_max_error":float((value-exact).abs().max()),"BOS_fixed":True,"zero_rhs_fixed":True})
    return {"passed":True,"seed":200060,"dtype":"float64","length":length,"width":width,"layers":2,
      "earlier_to_later_block_norm":coupling,"derivatives":checks,"linear_checks":linear,
      "scope":"tiny synthetic full causal algebra; actual prefix and reconstruction unqualified"}
if __name__=="__main__":print(json.dumps(reference()))
