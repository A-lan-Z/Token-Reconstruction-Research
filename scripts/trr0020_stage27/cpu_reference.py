"""Compare cached current-token algebra with independent full-sequence autograd."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
for path in ["scripts/trr0020_stage21","scripts/trr0020_stage23","scripts/trr0020_stage24"]:
    sys.path.insert(0,str(ROOT/path))
import torch
from torch.nn import functional as F
from linearized_prefix import forward as full_forward
from single_position import forward,jvp,vjp,commit
from normalized import equation
from stable_cgls import least_squares

def reference():
    torch.set_num_threads(2);torch.manual_seed(200064);dt=torch.float64;length,width,dim=4,8,4
    layers=[]
    for _ in range(2):
        p={key:.12*torch.randn(out,inp,dtype=dt) for key,out,inp in
          [('q',8,8),('k',4,8),('v',4,8),('o',8,8),('gate',12,8),('up',12,8),('down',8,12)]}
        p.update(n1=1+.1*torch.randn(width,dtype=dt),n2=1+.1*torch.randn(width,dtype=dt),
          eps1=1e-5,eps2=1e-5,heads=2,kv_heads=1,head_dim=dim,groups=2,scale=dim**-.5)
        layers.append(p)
    phase=torch.randn(length,dim//2,dtype=dt);phase=torch.cat([phase,phase],-1);cos,sin=phase.cos(),phase.sin()
    x=torch.randn(length,width,dtype=dt);expected,_=full_forward(x,layers,cos,sin)
    past=[(torch.empty(1,0,dim,dtype=dt),torch.empty(1,0,dim,dtype=dt)) for _ in layers]
    forwards=[];derivatives=[];linear=[]
    for pos in range(length):
        cur=x[pos:pos+1];co=cos[pos:pos+1];si=sin[pos:pos+1]
        actual,caches,current=forward(cur,layers,past,co,si)
        torch.testing.assert_close(actual,expected[pos:pos+1],rtol=1e-12,atol=1e-12)
        forwards.append({'position':pos,'max_error':float((actual-expected[pos:pos+1]).abs().max())})
        target=actual+.1*torch.randn_like(actual)
        def full_ref(value):return full_forward(torch.cat([x[:pos],value],dim=0),layers,cos[:pos+1],sin[:pos+1])[0][-1:]
        for mode in ['raw','normalized']:
            if mode=='raw':ref=full_ref;mv=lambda t:jvp(t,layers,caches,co,si);rmv=lambda t:vjp(t,layers,caches,co,si);b=target-actual
            else:
                ref=lambda v:F.normalize(full_ref(v),dim=-1)
                b,mv,rmv=equation(actual,target,lambda t:jvp(t,layers,caches,co,si),lambda t:vjp(t,layers,caches,co,si))
            dense=torch.autograd.functional.jacobian(ref,cur).reshape(width,width)
            for rep in range(3):
                a=torch.randn_like(cur);bvec=torch.randn_like(cur);ja=mv(a);jtb=rmv(bvec)
                exact_j=(dense@a.flatten())[None];exact_jt=(dense.T@bvec.flatten())[None]
                torch.testing.assert_close(ja,exact_j,rtol=1e-11,atol=1e-11)
                torch.testing.assert_close(jtb,exact_jt,rtol=1e-11,atol=1e-11)
                derivatives.append({'position':pos,'mode':mode,'rep':rep,'jvp_max_error':float((ja-exact_j).abs().max()),'vjp_max_error':float((jtb-exact_jt).abs().max())})
            direction,stats=least_squares(mv,rmv,b,32,.0001)
            exact=torch.linalg.solve(dense.T@dense+stats['ridge'][0]*torch.eye(width,dtype=dt),dense.T@b.flatten())[None]
            torch.testing.assert_close(direction,exact,rtol=1e-7,atol=1e-9)
            linear.append({'position':pos,'mode':mode,'dense_direction_max_error':float((direction-exact).abs().max()),'frozen_iteration':stats['frozen_iteration'].tolist(),'normal_relative_tolerance':stats['normal_relative_tolerance']})
        past=commit(past,current)
        assert all(old[0].shape[1]==pos+1 and old[1].shape[1]==pos+1 for old in past)
    return {'passed':True,'seed':200064,'dtype':'float64','forward_checks':forwards,'derivative_checks':derivatives,'linear_checks':linear,
      'scope':'tiny cached single-position algebra only; no GPU or reconstruction qualification'}
if __name__=='__main__':print(json.dumps(reference()))
