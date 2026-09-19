"""Independent tiny dense-Jacobian and dense-linear-solve references."""
import torch
from linearized_prefix import forward,diagonal_jvp,krylov_least_squares

def reference():
    torch.set_num_threads(2);torch.manual_seed(200048);dt=torch.float64
    length,width,heads,kv_heads,head_dim,intermediate=4,8,2,1,4,12
    layers=[]
    for _ in range(2):
        p={key:.12*torch.randn(out,inp,dtype=dt) for key,out,inp in [('q',8,8),('k',4,8),('v',4,8),('o',8,8),('gate',12,8),('up',12,8),('down',8,12)]}
        p.update(n1=1+.1*torch.randn(width,dtype=dt),n2=1+.1*torch.randn(width,dtype=dt),eps1=1e-5,eps2=1e-5,heads=heads,kv_heads=kv_heads,head_dim=head_dim,groups=heads//kv_heads,scale=head_dim**-.5);layers.append(p)
    phase=torch.randn(length,head_dim//2,dtype=dt);phase=torch.cat([phase,phase],dim=-1);cos=phase.cos();sin=phase.sin()
    x=torch.randn(length,width,dtype=dt)
    y,cache=forward(x,layers,cos,sin)
    jac=torch.autograd.functional.jacobian(lambda value:forward(value,layers,cos,sin)[0],x)
    blocks=torch.stack([jac[i,:,i,:] for i in range(length)])
    future=max(float(jac[i,:,j,:].abs().max()) for i in range(length) for j in range(i+1,length))
    assert future==0
    errors=[]
    for _ in range(5):
        tangent=torch.randn_like(x);actual=diagonal_jvp(tangent,layers,cache,cos,sin)
        expected=(blocks@tangent[...,None])[...,0]
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
        errors.append(float((actual-expected).abs().max()))
    b=.1*torch.randn_like(x);b[0]=0
    direction,stats=krylov_least_squares(lambda v:diagonal_jvp(v,layers,cache,cos,sin),b,steps=width,ridge=1e-10)
    exact=torch.linalg.solve(blocks,b[...,None])[...,0]
    torch.testing.assert_close(direction,exact,rtol=1e-7,atol=1e-9)
    assert torch.equal(direction[0],torch.zeros_like(direction[0]))
    zero,_=krylov_least_squares(lambda v:v,torch.zeros_like(b),steps=4,ridge=1e-6)
    assert torch.equal(zero,torch.zeros_like(zero))
    return {'passed':True,'seed':200048,'dtype':'float64','geometry':{'positions':length,'width':width,'layers':2,'query_heads':heads,'kv_heads':kv_heads},'jvp_max_errors':errors,'future_input_jacobian_max':future,'linear_direction_max_error':float((direction-exact).abs().max()),'linear_residual_norms':stats['residual_norm'].tolist(),'zero_rhs_preserved':True,'scope':'tiny synthetic algebra only; no real-prefix GPU qualification or reconstruction run'}
