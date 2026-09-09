"""One discrete trust schedule and one capped matrix-free inverse, no fitted starts."""
from shared import *
from executor import CurrentToken
from torch.nn.attention import sdpa_kernel,SDPBackend
import math

def reconstruct(prefix,observation,method,guard=None):
    if method=='cached':
        from cached import reconstruct as decode,Settings
        return decode(prefix,observation,Settings(optimizer='adam',learning_rate=.01),guard=guard)
    if method not in ['discrete','inverse']:raise ValueError(method)
    if any(p.requires_grad for p in prefix.parameters()):raise ValueError('frozen prefix required')
    if not torch.isfinite(observation).all():raise ValueError('nonfinite observation')
    sync();start=time.perf_counter()
    table=prefix.embed_tokens.weight.detach();E32=table.float()
    norms=E32.square().sum(-1)
    radius=2*norms.mean().sqrt()
    rng=torch.Generator(device='cpu').manual_seed(4401)
    tokens=[128000];traces=[];ex=CurrentToken(prefix)
    for pos in range(1,len(observation)):
        if time.perf_counter()-start>600:
            tokens.extend([-1]*(len(observation)-pos));break
        if guard:guard()
        sync();t0=time.perf_counter()
        target=observation[pos].to(table.device).float()
        initial=int(torch.randint(len(table),(1,),generator=rng))
        best=initial;best_loss=float('inf');checks=[];visited=torch.zeros(len(table),device=table.device,dtype=torch.bool)
        gradients=scans=jvps=vjps=continuous=0;reason='budget_exhausted';ties=0
        def verify(ids):
            nonlocal best,best_loss,reason
            for v in ids:
                with torch.no_grad():
                    y=ex(table[v]).float();loss=float((y-target).square().mean())
                if not math.isfinite(loss):raise RuntimeError('nonfinite candidate')
                checks.append({'token':v,'mse':loss});visited[v]=True
                if loss<best_loss:best,best_loss=v,loss
                if torch.allclose(y,target,atol=1e-5,rtol=1e-5):
                    reason='residual_accept';return True
            return False
        def rank(center):
            nonlocal scans,ties
            with torch.no_grad():
                d=norms-2*(E32@center)+center.square().sum();d.masked_fill_(visited,float('inf'))
                # Direct distances for a small shortlist; token IDs resolve true ties.
                idx=torch.topk(d,32,largest=False).indices
                exact=(E32[idx]-center).square().sum(-1)
                pairs=sorted(zip(exact.tolist(),idx.tolist()))
                scans+=1;ties+=sum(pairs[i][0]==pairs[i-1][0] for i in range(1,len(pairs)))
                return [v for _,v in pairs[:8]]
        accepted=verify([initial])
        z=table[initial].float().clone()
        lam=None
        for outer in range(4 if method=='inverse' else 8):
            if accepted:break
            if time.perf_counter()-t0>30:
                reason='token_timeout';break
            if method=='discrete':
                z=table[best].float().detach().clone().requires_grad_()
                y=ex(z).float();loss=.5*(y-target).square().mean()
                g,=torch.autograd.grad(loss,z);gradients+=1;continuous+=1
                if not torch.isfinite(g).all():raise RuntimeError('nonfinite gradient')
                if lam is None:lam=float(g.norm()/radius.clamp_min(1e-9))
                lam=max(lam,1e-8)
                center=z.detach()-g.detach()/lam
                ids=rank(center);before=best_loss;previous=z.detach()
                accepted=verify(ids)
                displacement=table[best].float()-previous
                predicted=-float(g@displacement+.5*lam*displacement.square().sum())
                achieved=.5*(before-best_loss)
                ratio=achieved/max(predicted,1e-12)
                # Actual improvement controls trust; no truth labels or score-gap certificate.
                if achieved<=0:lam*=.5
                elif ratio<.25:lam*=2
                elif ratio>.75:lam*=.5
            else:
                z=z.detach()
                width=z.numel()
                def fn(q):return ex(q).float()/math.sqrt(width)
                with sdpa_kernel(SDPBackend.MATH):
                    y,pb=torch.func.vjp(fn,z);continuous+=1
                    residual=y-target/math.sqrt(width);g=pb(residual)[0];vjps+=1;gradients+=1
                    if not torch.isfinite(g).all():raise RuntimeError('nonfinite inverse gradient')
                    _,jg=torch.func.jvp(fn,(z,),(g,));jvps+=1;continuous+=1
                    damp=max(.1*float(jg.square().sum()/g.square().sum().clamp_min(1e-20)),1e-5)
                    delta=torch.zeros_like(z);rr=-g;direction=rr.clone();rrnorm=rr@rr
                    for _ in range(4):
                        _,jd=torch.func.jvp(fn,(z,),(direction,));jvps+=1;continuous+=1
                        action=pb(jd)[0]+damp*direction;vjps+=1
                        alpha=rrnorm/(direction@action).clamp_min(1e-20)
                        delta=delta+alpha*direction;rrnew=rr-alpha*action
                        nn=rrnew@rrnew
                        direction=rrnew+(nn/rrnorm.clamp_min(1e-20))*direction
                        rr,rrnorm=rrnew,nn
                    old=float(residual.square().sum())
                    for factor in [1.,.5]:
                        trial=z+factor*delta
                        with torch.no_grad():new=float((fn(trial)-target/math.sqrt(width)).square().sum())
                        continuous+=1
                        if new<old:z=trial;break
                if not torch.isfinite(z).all():raise RuntimeError('nonfinite inverse step')
                accepted=verify(rank(z))
        tokens.append(best);ex.commit(best);sync()
        traces.append({'position':pos,'initial_token':initial,'token':best,'best_mse':best_loss,'reason':reason,'checks':checks,'gradient_steps':gradients,'jvps':jvps,'vjps':vjps,'continuous_forwards':continuous,'vocabulary_scans':scans,'vocabulary_rows_scored':scans*len(table),'shortlist_direct_distance_rows':scans*32,'near_ties':ties,'seconds':time.perf_counter()-t0})
    sync()
    return {'tokens':tokens,'trace':traces,'seconds':time.perf_counter()-start,'method':method,'learned_proposer':False,'cache':'immutable trial; single commit; fixed record weights','settings':{'seed':4401,'candidates_per_round':8,'rounds':4 if method=='inverse' else 8,'initialization':'one public random embedding','atol':1e-5,'rtol':1e-5,'candidate_batch':1,'cg_iterations':4 if method=='inverse' else 0}}
