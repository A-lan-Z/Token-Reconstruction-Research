"""Score all frozen development methods only after validating complete outputs."""
from native import *
import argparse
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--cells',type=int,required=True);a=p.parse_args()
assert (X/'fresh_score.json').exists()
receipt=json.loads((X/(a.name+'_freeze.json')).read_text())
assert len(receipt['entries'])==a.cells
assert len({(r['id'],r.get('method','')) for r in receipt['entries']})==a.cells
for row in receipt['entries']:
    if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('prediction changed')
truth=json.loads((OUT/'fresh_r1/evaluator_truth.json').read_text())
rows=[]
for row in receipt['entries']:
    d=load_file(str(ROOT/row['path']));t=torch.tensor(truth[row['id']]);correct=d['tokens'][1:]==t[1:];hits=(d['candidates']==t[1:,None]).any(-1)
    rows.append({**row,'correct':int(correct.sum()),'scored':len(t)-1,'exact':bool(correct.all()),'hits':int(hits.sum())})
summary=[]
for m in sorted({r.get('method','') for r in rows}):
 for c in ['matched','lora256']:
  for g in ['natural','stress']:
    rr=[r for r in rows if r['condition']==c and r['group']==g and r.get('method','')==m]
    summary.append({'method':m,'condition':c,'group':g,'correct':sum(r['correct'] for r in rr),'scored':sum(r['scored'] for r in rr),'exact':sum(r['exact'] for r in rr),'records':len(rr),'hits':sum(r['hits'] for r in rr),'seconds':sum(r['seconds'] for r in rr)})
write(X/(a.name+'_score.json'),{'summary':summary,'rows':rows,'scope':'retrospective development, not fresh confirmation'})
print(json.dumps(summary,indent=2),flush=True)
