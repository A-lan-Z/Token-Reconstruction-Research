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
            if engine.kind=="white":
                dual=g@engine.rinv.T;metric_z=(z@engine.r)@engine.r.T
            else:dual=g;metric_z=z
            curvature=(engine.multiplier*engine.curvature[:length-1]).clamp(1e-6,1e6)
            query=curvature[:,None]*metric_z-g
            scores=torch.mm(query.to(torch.bfloat16),engine.weight.T,out_dtype=torch.float32)-.5*curvature[:,None]*engine.norm
            ids[1:]=scores.argmax(-1)
            z.copy_(engine.weight[ids[1:]].float())
    outputs["best_objective"]=best.cpu();outputs["best_position_error"]=position.cpu()
    return outputs,trace
