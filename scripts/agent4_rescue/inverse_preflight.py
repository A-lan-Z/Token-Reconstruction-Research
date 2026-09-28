"""Adjoint and largest-search workspace checks, public synthetic only."""
from shared import *
from executor import CurrentToken
from torch.nn.attention import sdpa_kernel,SDPBackend
torch.set_num_threads(2);p=load_prefix(torch.bfloat16);ex=CurrentToken(p)
for i in range(126):ex.commit([2028,374,264,1296][i%4])
z=p.embed_tokens.weight[13].float().detach()
torch.manual_seed(9472);v=torch.randn_like(z);u=torch.randn_like(z)
normal=ex(z).detach().float()
with sdpa_kernel(SDPBackend.MATH):
    mathy,jv=torch.func.jvp(lambda q:ex(q).float(),(z,),(v,))
    _,pb=torch.func.vjp(lambda q:ex(q).float(),z)
    jtu=pb(u)[0]
left=float(jv@u);right=float(v@jtu)
adj=abs(left-right)/max(float(jv.norm()*u.norm()),float(v.norm()*jtu.norm()),1e-12)
assert adj<.01
scoring=p.embed_tokens.weight.float();norms=scoring.square().sum(-1);rank=norms-2*(scoring@z)+z.square().sum()
assert torch.isfinite(rank).all();guard()
write(E/'inverse_preflight.json',{'environment':environment(),'position':127,'adjoint_left':left,'adjoint_right':right,'normalized_adjoint_error':adj,'math_vs_sdpa_relative_output_error':float((mathy-normal).norm()/normal.norm()),'math_vs_sdpa_equal':torch.equal(mathy,normal),'peak_reserved':torch.cuda.max_memory_reserved(),'qualified':True,'budget':'4 local steps, each 4 CG iterations plus curvature estimate, <=20 JVPs/20 VJPs per token; no dense Jacobian'})
print('Derivative adjoint and largest workspace passed',adj,torch.cuda.max_memory_reserved())
