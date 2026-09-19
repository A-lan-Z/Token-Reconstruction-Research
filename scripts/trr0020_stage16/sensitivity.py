"""Prototype: estimate normalized-prefix sensitivity without a target residual."""
import torch
from torch.nn import functional as F
from discrete_parallel import DiscreteParallel

def estimate_curvature(engine,initial,probes=8,seed=200042):
    length=len(initial);pe,mask=engine.geometry[length]
    z=engine.weight[initial.to("cuda")[1:]].float().detach().requires_grad_()
    predicted=DiscreteParallel.diagonal_forward(engine,torch.cat([engine.bos,z[None]],dim=1),pe,mask)
    normalized=F.normalize(predicted[:,1:],dim=-1)
    generator=torch.Generator().manual_seed(seed)
    # Generate maximum-size fixed signs, then slice: native length does not alter the probe identity.
    signs=(torch.randint(0,2,(probes,1,127,2048),generator=generator)*2-1).to(device="cuda",dtype=torch.float32)
    estimates=[]
    for k in range(probes):
        g=torch.autograd.grad((normalized*signs[k,:,:length-1]).sum(),z,retain_graph=k+1<probes)[0]
        if engine.kind=="white":g=g@engine.rinv.T
        estimates.append(g.square().mean(-1))
    result=torch.stack(estimates).mean(0).detach()
    if not torch.isfinite(result).all() or not bool((result>0).all()):
        raise RuntimeError("nonpositive or nonfinite prefix sensitivity")
    return result

def sensitivity_step(self,length):
    self.gradient.zero_();self.evaluate(length).backward()
    with torch.no_grad():
        z=self.z[:length-1];g=self.gradient[:length-1]
        metric_z=(z@self.r)@self.r.T if self.kind=="white" else z
        curvature=(self.multiplier*self.curvature[:length-1]).clamp(1e-6,1e6)
        query=curvature[:,None]*metric_z-g
        scores=torch.mm(query.to(torch.bfloat16),self.weight.T,out_dtype=torch.float32)-.5*curvature[:,None]*self.norm
        ids=scores.argmax(-1)
        self.tokens[1:length].copy_(ids);self.z[:length-1].copy_(self.weight[ids].float())
