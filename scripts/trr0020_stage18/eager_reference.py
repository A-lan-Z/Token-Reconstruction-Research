import torch
from torch.nn import functional as F
from discrete_parallel import DiscreteParallel
SNAPSHOTS={0,1,2,4,8,16,32}

def eager_reference(engine,h,initial):
    """Independent eager loop, duplicating the declared score without train_step."""
    length=len(h);pe,mask=engine.geometry[length]
    target=h.to('cuda').float()[None,1:];ids=initial.to('cuda').clone()
    z=engine.weight[ids[1:]].float().detach().requires_grad_()
    best=ids.clone();position=ids.clone();best_loss=float('inf')
    position_loss=torch.full((length-1,),float('inf'),device='cuda');outputs={};trace=[]
    for step in range(33):
        if step in SNAPSHOTS:outputs['step'+str(step)]=ids.cpu()
        predicted=DiscreteParallel.diagonal_forward(engine,torch.cat([engine.bos,z[None]],dim=1),pe,mask)
        error=(1-F.cosine_similarity(predicted[:,1:],target,dim=-1))[0]
        value=float(error.detach().mean());trace.append(value)
        if value<best_loss:best=ids.clone();best_loss=value
        with torch.no_grad():
            better=error<position_loss;position[1:]=torch.where(better,ids[1:],position[1:])
            position_loss=torch.minimum(position_loss,error)
        if step==32:break
        g=torch.autograd.grad(error.sum(),z)[0]
        with torch.no_grad():
            loss=error.clamp_min(1e-8)
            if engine.kind=='white':dual=g@engine.rinv.T;metric_z=(z@engine.r)@engine.r.T
            else:dual=g;metric_z=z
            lam=(engine.beta*dual.square().sum(-1)/(2*loss)).clamp(1e-6,1e6)
            linear=g@engine.score_weight.T-(g*z).sum(-1,keepdim=True)
            distance=(engine.norm[None]+(z@engine.r).square().sum(-1,keepdim=True)-2*(metric_z@engine.score_weight.T)).clamp_min(0)
            scores=-linear-.5*lam[:,None]*distance
            if engine.gamma:scores=scores-engine.gamma*linear.square()/(4*loss[:,None])
            ids[1:]=scores.argmax(-1);z.copy_(engine.weight[ids[1:]].float())
    outputs['best_objective']=best.cpu();outputs['best_position_error']=position.cpu()
    return outputs,trace

def score_reference(engine,h,initial):
    """Sample the actual full-vocabulary GPU calculation against float64 delta math."""
    length=len(h);pe,mask=engine.geometry[length]
    z=engine.weight[initial.to('cuda')[1:]].float().detach().requires_grad_()
    predicted=DiscreteParallel.diagonal_forward(engine,torch.cat([engine.bos,z[None]],dim=1),pe,mask)
    error=(1-F.cosine_similarity(predicted[:,1:],h.to('cuda').float()[None,1:],dim=-1))[0]
    g=torch.autograd.grad(error.sum(),z)[0]
    with torch.no_grad():scores=engine.vocabulary_scores(z,g,error)
    pos=torch.tensor([0,(length-1)//2,length-2],device='cuda')
    tok=torch.tensor([0,7,len(engine.weight)//2,len(engine.weight)-1],device='cuda')
    zd=z[pos].detach().cpu().double();gd=g[pos].detach().cpu().double()
    loss=error[pos].detach().cpu().double().clamp_min(1e-8)
    e=engine.score_weight[tok].detach().cpu().double();r=engine.r.cpu().double();ri=engine.rinv.cpu().double()
    delta=e[None]-zd[:,None];linear=(gd[:,None]*delta).sum(-1)
    lam=(engine.beta*(gd@ri.T).square().sum(-1)/(2*loss)).clamp(1e-6,1e6)
    exact=-linear-.5*lam[:,None]*(delta@r).square().sum(-1)-engine.gamma*linear.square()/(4*loss[:,None])
    actual=scores[pos[:,None],tok[None]].cpu().double();difference=(actual-exact).abs()
    return {'positions':pos.cpu().tolist(),'tokens':tok.cpu().tolist(),'actual':actual.tolist(),
      'float64_reference':exact.tolist(),'maximum_absolute_error':float(difference.max()),
      'maximum_relative_error':float((difference/exact.abs().clamp_min(1e-12)).max()),
      'rtol':2e-4,'atol':5e-6,'all_close':bool(torch.allclose(actual,exact,rtol=2e-4,atol=5e-6))}
