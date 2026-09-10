from __future__ import annotations
import json, time, math, platform, hashlib
from pathlib import Path
import torch
from prototype import TinyPrefix, window_search, DTYPE

ROOT=Path(__file__).parent

def analytic_checks():
    # Local continuous freedom can absorb all future evidence. y=(x^3,x+u).
    x=torch.tensor(0.,dtype=DTYPE,requires_grad=True)
    u=torch.tensor(0.,dtype=DTYPE,requires_grad=True)
    current=0.5*(x**3-1)**2
    joint=current+0.5*(x+u-1)**2
    gc=torch.autograd.grad(current,x,retain_graph=True)[0]
    gj=torch.autograd.grad(joint,x)[0]
    # Profile u=1-x. The whole future residual is zero at each x.
    # Restrict u to {0,1}; x=0 remains stationary in the profiled objective.
    # Crucial qualification: an arbitrary wrong tail can generate gradients that
    # look helpful; profiling can remove them. Discrete constraints not magic.
    discrete_states=[]
    for xv in [0.,.25,.5,.75,1.]:
        vals=[.5*(xv**3-1)**2+.5*(xv+uv-1)**2 for uv in [0.,1.]]
        discrete_states.append({'x':xv,'continuous_profile':.5*(xv**3-1)**2,
                                'discrete_profile':min(vals)})
    # Constructed stronger cross-token channel: y=(x^3, 2*x+u), truth (1,0).
    # Under finite vocabulary {0,1}, neither possible u cancels at wrong root 0.
    x2=torch.tensor(0.,dtype=DTYPE,requires_grad=True)
    grads=[]
    for uv in [0.,1.]:
        obj=.5*(x2**3-1)**2+.5*(2*x2+uv-2)**2
        grads.append(float(torch.autograd.grad(obj,x2,retain_graph=True)[0]))
    vals=[(.5*(a**3-1)**2+.5*(2*a+b-2)**2,a,b) for a in [0.,1.] for b in [0.,1.]]
    # Linear triangular cancellation with full future embedding freedom.
    g=torch.Generator().manual_seed(9803)
    d=8; A=torch.randn(d,d,generator=g,dtype=DTYPE); A=A+4*torch.eye(d,dtype=DTYPE)
    B=torch.randn(d,d,generator=g,dtype=DTYPE); C=torch.randn(d,d,generator=g,dtype=DTYPE)+4*torch.eye(d,dtype=DTYPE)
    delta=torch.randn(d,generator=g,dtype=DTYPE)
    future=torch.linalg.solve(C,-B@delta)
    cancellation=float(torch.linalg.vector_norm(B@delta+C@future))
    # Explicit Jacobian nuisance projection I-CC^+ -> zero.
    proj=torch.eye(d,dtype=DTYPE)-C@torch.linalg.pinv(C)
    return {'single_observation_zero_gradient':float(gc),'joint_gradient_with_fixed_wrong_tail':float(gj),
            'unit_coupling_profile':discrete_states,
            'stronger_coupling_root_gradients_for_all_discrete_tails':grads,
            'stronger_coupling_exact_best_block':sorted(vals)[0],
            'linear_future_cancellation_norm':cancellation,
            'projected_cross_token_jacobian_norm':float(torch.linalg.matrix_norm(proj@B)),
            'interpretation':'Constructed analytic examples, not empirical LM observations.'}

def cache_checks():
    m=TinyPrefix(802); ids=torch.tensor([0,19,32,76,4,21,9])
    # Cached versus full differentiable forward at prefix length3, block4.
    cache=m.commit(ids[:3],m.empty_cache())
    old=[(k.clone(),v.clone()) for k,v in cache]
    z=m.E[ids[3:]][None].clone().requires_grad_(True)
    y=m.forward(z,cache)
    weight=torch.linspace(.2,1.,y.numel(),dtype=DTYPE).reshape(y.shape)
    g1=torch.autograd.grad((y*weight).sum(),z)[0]
    z2=m.E[ids[3:]][None].clone().requires_grad_(True)
    y2=m.forward(torch.cat([m.E[ids[:3]][None],z2],1))[:,3:]
    g2=torch.autograd.grad((y2*weight).sum(),z2)[0]
    immutable=all(torch.equal(k,k0) and torch.equal(v,v0) for (k,v),(k0,v0) in zip(cache,old))
    assert immutable
    assert torch.allclose(y,y2,atol=1e-10,rtol=1e-10)
    assert torch.allclose(g1,g2,atol=1e-10,rtol=1e-10)
    return {'output_max_abs_error':float((y-y2).abs().max()),'gradient_max_abs_error':float((g1-g2).abs().max()),'cache_immutable':immutable}

def main():
    torch.set_num_threads(1); torch.manual_seed(801)
    results={'scope':'Synthetic random causal transformer, CPU FP64, no fitting; NOT Llama.',
             'environment':{'torch':torch.__version__,'python':platform.python_version(),'threads':1},
             'analysis':analytic_checks(),'cache':cache_checks(),'config':{'seeds':[100,101,102],
             'sequences_per_seed':16,'unknown_tokens_per_sequence':8,'vocab':128,'hidden':32,
             'layers':4,'window':3,'beam':2,'proposal_count':2,'rounds_cap':4},'rows':[]}
    start=time.perf_counter()
    modes=['sequential','diagonal','joint']
    for s in results['config']['seeds']:
        m=TinyPrefix(s)
        gen=torch.Generator().manual_seed(s+10000)
        for r in range(16):
            truth=torch.randint(1,128,(8,),generator=gen)
            seq=torch.cat([torch.tensor([0]),truth])
            with torch.no_grad(): obs=m.forward(m.E[seq][None])[0,1:].clone()
            # Counterbalance wall-time order. The inference routine is not given truth.
            order=modes[r%3:]+modes[:r%3]
            for mode in order:
                out=window_search(m,obs,mode=mode)
                out.update(seed=s,record=r,method=mode,truth=truth.tolist())
                out['correct']=sum(a==b for a,b in zip(out['prediction'],truth.tolist()))
                out['exact']=out['correct']==8
                results['rows'].append(out)
    results['total_seconds']=time.perf_counter()-start
    agg={}
    for mode in modes:
        rr=[x for x in results['rows'] if x['method']==mode]
        agg[mode]={'correct':sum(x['correct'] for x in rr),'tokens':8*len(rr),'exact':sum(x['exact'] for x in rr),'sequences':len(rr),
                   'cpu_seconds':sum(x['seconds'] for x in rr),
                   'work':{k:sum(x['work'][k] for x in rr) for k in rr[0]['work']},
                   'per_seed':{str(s):{'correct':sum(x['correct'] for x in rr if x['seed']==s),'exact':sum(x['exact'] for x in rr if x['seed']==s)} for s in [100,101,102]}}
    results['summary']=agg
    by={(x['seed'],x['record'],x['method']):x for x in results['rows']}
    paired={}
    for control in ['sequential','diagonal']:
        g=l=bg=bl=0
        for s in [100,101,102]:
            for r in range(16):
                a=by[s,r,'joint']; b=by[s,r,control]; tru=a['truth']
                g+=sum(x==t and y!=t for x,y,t in zip(a['prediction'],b['prediction'],tru))
                l+=sum(x!=t and y==t for x,y,t in zip(a['prediction'],b['prediction'],tru))
                bg+=a['exact'] and not b['exact']; bl+=b['exact'] and not a['exact']
        paired[control]={'token_gains':g,'token_losses':l,'exact_gains':int(bg),'exact_losses':int(bl)}
    results['paired_joint_vs']=paired
    for p in ['PLAN.md','prototype.py',Path(__file__).name]:
        results.setdefault('source_sha256',{})[p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    (ROOT/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'analysis':results['analysis'],'cache':results['cache'],'summary':agg,'paired':paired,'total_seconds':results['total_seconds']},indent=2))
if __name__=='__main__': main()
