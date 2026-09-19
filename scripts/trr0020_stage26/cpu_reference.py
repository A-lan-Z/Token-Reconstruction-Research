"""Independent dense checks of right scaling and regularized global solve."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage25"))
import torch
from precondition import make_scale,scaled_operators,KINDS
from global_cgls import least_squares

def reference():
    torch.set_num_threads(2);torch.manual_seed(200062);dtype=torch.float64
    length,width=4,6;n=length*width
    a=torch.randn((n,n),dtype=dtype)*.15+torch.eye(n,dtype=dtype)
    a=a*torch.logspace(-1,1,n,dtype=dtype);a[:width]=0;a[:,:width]=0
    x=torch.randn(length,width,dtype=dtype)
    probes=[(torch.randint(0,2,(length,width))*2-1).to(dtype) for _ in range(4)]
    mv=lambda v:(a@v.flatten()).reshape_as(v)
    rmv=lambda v:(a.T@v.flatten()).reshape_as(v)
    b=torch.randn_like(x);b[0]=0;checks=[]
    energy=torch.stack([(a.T@v.flatten()).square().reshape_as(x) for v in probes]).mean(0)
    for kind in KINDS:
        scale,work=make_scale(kind,x,rmv,probes)
        if kind=="identity":expected=torch.ones_like(x)
        elif kind=="input_norm":expected=x.norm(dim=-1,keepdim=True).expand_as(x)
        else:
            position=energy.mean(-1,keepdim=True);floor=position[1:].median()*1e-6
            diagonal=position.expand_as(x) if kind=="position_probe4" else .5*energy+.5*position
            expected=diagonal.clamp_min(floor).rsqrt()
        expected=expected/expected[1:].median();expected[0]=0
        torch.testing.assert_close(scale,expected,rtol=1e-12,atol=1e-12)
        j,jt=scaled_operators(mv,rmv,scale);dense=a*scale.flatten()[None]
        left=torch.randn_like(x);right=torch.randn_like(x)
        torch.testing.assert_close(j(right).flatten(),dense@right.flatten(),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(jt(left).flatten(),dense.T@left.flatten(),rtol=1e-12,atol=1e-12)
        for damping in [.0001,.01]:
            value,stats=least_squares(j,jt,b,128,damping)
            exact=torch.linalg.solve(dense.T@dense+stats["ridge"]*torch.eye(n,dtype=dtype),dense.T@b.flatten()).reshape_as(x)
            torch.testing.assert_close(value,exact,rtol=1e-7,atol=1e-9)
            delta=scale*value
            torch.testing.assert_close(stats["recurrence_residual_norm"],(mv(delta)-b).norm(dim=-1),rtol=1e-6,atol=1e-10)
            assert torch.equal(delta[0],torch.zeros_like(delta[0]))
            zero,_=least_squares(j,jt,torch.zeros_like(b),8,damping)
            assert torch.equal(zero,torch.zeros_like(zero))
            checks.append({"kind":kind,"damping":damping,"scale_max_error":float((scale-expected).abs().max()),
              "dense_direction_max_error":float((value-exact).abs().max()),"BOS_fixed":True,"zero_rhs_fixed":True,**work})
    return {"passed":True,"seed":200062,"dtype":"float64","geometry":[length,width],"checks":checks,
      "scope":"synthetic scaling and global recurrence only; not reconstruction evidence"}
if __name__=="__main__":print(json.dumps(reference()))
