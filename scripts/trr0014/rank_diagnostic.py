"""Post-score proposal-rank diagnostic; never an inference result."""
from native import *
from token_reconstruction.prefix_weight_metric import PrefixWeightMetric
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
assert (X/'fresh_score.json').exists()
prefix=load_prefix();metric=PrefixWeightMetric(prefix);stats=metric.build();guard()
panel=OUT/'fresh_r1';obs=load_file(str(panel/'observations.safetensors'));meta=json.loads((panel/'metadata.json').read_text())
# This panel is now opened development. Nothing from this diagnostic changes its frozen results.
truth=json.loads((panel/'evaluator_truth.json').read_text())
rows=[]
for r in meta:
    ids=metric.propose(obs[r['id']],2048).cpu();t=torch.tensor(truth[r['id']]);equal=ids[1:]==t[1:,None]
    found=equal.any(-1);rank=equal.to(torch.int64).argmax(-1)+1;rank[~found]=128257
    rows.append({**r,'ranks':rank.tolist(),'recall_counts':{str(k):int((rank<=k).sum()) for k in [1,8,16,32,64,128,256,512,1024,2048]},'scored':len(t)-1})
summary=[]
for c in ['matched','lora256']:
    for g in ['natural','stress']:
        rr=[r for r in rows if r['condition']==c and r['group']==g];n=sum(r['scored'] for r in rr)
        summary.append({'condition':c,'group':g,'scored':n,'hits':{str(k):sum(r['recall_counts'][str(k)] for r in rr) for k in [1,8,16,32,64,128,256,512,1024,2048]}})
write(X/'opened_rank_diagnostic.json',{'environment':environment(),'scope':'post-score proposal diagnostic on opened panel; no changed reconstructions','summary':summary,'rows':rows})
print(json.dumps(summary,indent=2),flush=True)
