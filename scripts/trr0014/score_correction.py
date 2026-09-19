from native import *
receipt=json.loads((X/'correction_dev_r1_freeze.json').read_text())
for row in receipt['entries']:
    if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('prediction changed')
assert len(receipt['entries'])==48
truth=json.loads((OUT/'fresh_r1/evaluator_truth.json').read_text())
rows=[]
for row in receipt['entries']:
    d=load_file(str(ROOT/row['path']));t=torch.tensor(truth[row['id']]);correct=d['tokens'][1:]==t[1:];hits=(d['candidates']==t[1:,None]).any(-1)
    rows.append({**row,'correct':int(correct.sum()),'scored':len(t)-1,'exact':bool(correct.all()),'hits':int(hits.sum())})
summary=[]
for c in ['matched','lora256']:
 for g in ['natural','stress']:
    rr=[r for r in rows if r['condition']==c and r['group']==g]
    summary.append({'condition':c,'group':g,'correct':sum(r['correct'] for r in rr),'scored':sum(r['scored'] for r in rr),'exact':sum(r['exact'] for r in rr),'records':len(rr),'hits':sum(r['hits'] for r in rr),'seconds':sum(r['seconds'] for r in rr)})
write(X/'correction_dev_r1_score.json',{'summary':summary,'rows':rows,'scope':'retrospective development, not fresh confirmation'})
print(json.dumps(summary,indent=2),flush=True)
