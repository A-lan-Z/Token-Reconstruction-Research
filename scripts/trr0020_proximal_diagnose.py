"""Public-only diagnosis of direct quadratic updates; no development labels."""
from pathlib import Path
import sys,json,time,os
ROOT=Path(__file__).resolve().parents[1]
for path in ["scripts/trr0014","scripts/trr0020","scripts/trr0017","scripts/trr0020_stage10"]:
    sys.path.insert(0,str(ROOT/path))
import native as n
from proximal import ProximalVocabulary
torch=n.torch;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(True)
X=ROOT/"experiments/TRR-0020"
prefix=n.load_prefix()
generator=torch.Generator().manual_seed(200030)
ids=torch.randint(256,128000,(1,128),generator=generator).to("cuda");ids[0,0]=128000
h=prefix.forward_full(ids)[0].float()
rows=[]
for metric in ["raw","white"]:
    engine=ProximalVocabulary(prefix,metric,.25)
    engine.ensure(128);engine.reset(h);engine.gradient.zero_()
    loss=engine.evaluate(128);loss.backward()
    z=engine.z.detach();g=engine.gradient
    n.guard()
    dual=g@engine.rinv.T if metric=="white" else g
    metric_z=(z@engine.r)@engine.r.T if metric=="white" else z
    error=engine.error.detach()
    true=ids[0,1:]
    for beta in [.0001,.001,.01,.1,.25]:
        lam=(beta*dual.square().sum(-1)/(2*error.clamp_min(1e-8))).clamp(1e-6,1e6)
        query=lam[:,None]*metric_z-g
        scores=torch.mm(query.to(torch.bfloat16),engine.weight.T,out_dtype=torch.float32)-.5*lam[:,None]*engine.norm
        predicted=scores.argmax(-1)
        true_score=scores.gather(1,true[:,None])[:,0]
        current_score=scores.gather(1,engine.tokens[1:,None])[:,0]
        def quantile(v):return torch.quantile(v.float(),torch.tensor([0.,.25,.5,.75,1.],device="cuda")).cpu().tolist()
        rows.append({"metric":metric,"beta":beta,"current_correct":int((engine.tokens[1:]==true).sum()),
          "one_update_correct":int((predicted==true).sum()),"changed_positions":int((predicted!=engine.tokens[1:]).sum()),
          "gradient_norm_quantiles":quantile(g.norm(dim=-1)),"embedding_norm_quantiles":quantile(z.norm(dim=-1)),
          "error_quantiles":quantile(error),"curvature_quantiles":quantile(lam),
          "true_local_score_better_than_current":int((true_score>current_score).sum()),
          "grad_same_storage":engine.z.grad.data_ptr()==engine.gradient.data_ptr()})
    del engine
    __import__("gc").collect();torch.cuda.empty_cache()
n.write(X/"proximal_public_diagnostic.json",{"environment":n.environment(),"source_sha256":n.digest(Path(__file__)),
 "implementation_sha256":n.digest(ROOT/"scripts/trr0020_stage10/proximal.py"),"seed":200030,"scope":"public random synthetic diagnosis","rows":rows})
print(json.dumps(rows,indent=2),flush=True)
