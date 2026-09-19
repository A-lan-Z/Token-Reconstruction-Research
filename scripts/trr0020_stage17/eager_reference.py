import torch
from torch.nn import functional as F
from discrete_parallel import DiscreteParallel
SNAPSHOTS={0,1,2,4,8,16,32}

def eager_reference(engine,h,initial):
    """Independent eager control flow for the direct stage; same declared prefix arithmetic."""
    length=len(h);pe,mask=engine.geometry[length]
    target=h.to("cuda").float()[None,1:]
    ids=initial.to("cuda").clone()
    z=engine.weight[ids[1:]].float().detach().requires_grad_()
    best=ids.clone();position=ids.clone()
    best_loss=float("inf");position_loss=torch.full((length-1,),float("inf"),device="cuda")
    outputs={};trace=[]
    for step in range(33):
        if step in SNAPSHOTS:outputs["step"+str(step)]=ids.cpu()
        full=torch.cat([engine.bos,z[None]],dim=1)
        predicted=DiscreteParallel.diagonal_forward(engine,full,pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],target,dim=-1))[0]
        value=float(error.detach().mean());trace.append(value)
        if value<best_loss:best=ids.clone();best_loss=value
        with torch.no_grad():
            better=error<position_loss
            position[1:]=torch.where(better,ids[1:],position[1:])
            position_loss=torch.minimum(position_loss,error)
        if step==32:break
        g=torch.autograd.grad(error.sum(),z)[0]
        with torch.no_grad():
            count=length-1
            metric_z=(z@engine.r)@engine.r.T if engine.kind=="white" else z
            bz=engine.alpha*engine.curvature[:count,None]*metric_z
            if engine.alpha<1:
                probes=engine.probes[:,:count]
                bz=bz+(1-engine.alpha)*((probes*z[None]).sum(-1)[...,None]*probes).mean(0)
            scores=(engine.scale*bz-g)@engine.score_weight.T-.5*engine.scale*engine.quadratic_norm[:count]
            ids[1:]=scores.argmax(-1)
            z.copy_(engine.weight[ids[1:]].float())
    outputs["best_objective"]=best.cpu();outputs["best_position_error"]=position.cpu()
    return outputs,trace


def cache_reference(engine,length):
    """Check sampled cached entries against separate CPU float64 products."""
    positions=torch.tensor([0,(length-1)//2,length-2],device="cuda")
    tokens=torch.tensor([0,7,len(engine.weight)//2,len(engine.weight)-1],device="cuda")
    embedding=engine.score_weight[tokens].detach().cpu().double()
    metric=engine.r.detach().cpu().double()
    probes=engine.probes[:,positions].detach().cpu().double()
    curvature=engine.curvature[positions].detach().cpu().double()
    exact=engine.alpha*curvature[:,None]*(embedding@metric).square().sum(-1)[None]
    exact+=(1-engine.alpha)*(probes@embedding.T).square().mean(0)
    actual=engine.quadratic_norm[positions[:,None],tokens[None]].detach().cpu().double()
    difference=(actual-exact).abs()
    return {"positions":positions.cpu().tolist(),"tokens":tokens.cpu().tolist(),
      "actual":actual.tolist(),"float64_reference":exact.tolist(),
      "maximum_absolute_error":float(difference.max()),
      "maximum_relative_error":float((difference/exact.abs().clamp_min(1e-12)).max()),
      "rtol":1e-4,"atol":1e-6,"all_close":bool(torch.allclose(actual,exact,rtol=1e-4,atol=1e-6))}
