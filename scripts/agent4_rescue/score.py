"""Evaluator only: all compared outputs must exist and pass hash checks first."""
from shared import *
import argparse,numpy as np
p=argparse.ArgumentParser();p.add_argument('--panel',default='old');p.add_argument('--runs',nargs='+',required=True);p.add_argument('--name',required=True);a=p.parse_args()
freezes={n:json.loads((E/n/'freeze.json').read_text()) for n in a.runs}
for n,f in freezes.items():
    for item in f['prediction_files']:assert digest(ROOT/item['path'])==item['sha256']
root=OLD/'gpu_retrospective/bfloat16' if a.panel=='old' else OUT/a.panel
truth=json.loads((root/'evaluator_truth.json').read_text())
results={}
for n,f in freezes.items():
    records=[]
    for item in f['prediction_files']:
        row=json.loads((ROOT/item['path']).read_text());gold=truth[row['record_id']];pred=row['tokens'];assert len(pred)==len(gold)
        positions=[]
        for t in row['trace']:
            i=t['position'];found=any(c['token']==gold[i] for c in t['checks'])
            positions.append({'position':i,'correct':pred[i]==gold[i],'discovered':found,'reason':t['reason'],'checks':len(t['checks']),'gradient_steps':t['gradient_steps'],'vocabulary_scans':t['vocabulary_scans'],'seconds':t['seconds'],'prior_error':any(x!=y for x,y in zip(pred[:i],gold[:i]))})
        records.append({'record_id':row['record_id'],'group':row['group'],'correct':sum(x==y for x,y in zip(pred[1:],gold[1:])),'denominator':len(gold)-1,'exact':pred==gold,'seconds':row['seconds'],'positions':positions})
    groups={}
    for group in ['ordinary','stress','all']:
        rr=[r for r in records if group=='all' or r['group']==group]
        if not rr:continue
        pp=[p for r in rr for p in r['positions']]
        groups[group]={'correct':sum(r['correct'] for r in rr),'denominator':sum(r['denominator'] for r in rr),'exact':sum(r['exact'] for r in rr),'records':len(rr),'seconds':sum(r['seconds'] for r in rr),'median_seconds':float(np.median([r['seconds'] for r in rr])),'p95_seconds':float(np.quantile([r['seconds'] for r in rr],.95)),'checks':sum(p['checks'] for p in pp),'gradients':sum(p['gradient_steps'] for p in pp),'scans':sum(p['vocabulary_scans'] for p in pp),'exhaustions':sum(p['reason']=='budget_exhausted' for p in pp),'timeouts':sum('timeout' in p['reason'] for p in pp),'missed':sum(not p['discovered'] for p in pp)}
    results[n]={'groups':groups,'records':records,'cost':f}
write(E/(a.name+'.json'),{'results':results,'truth_sha256':digest(root/'evaluator_truth.json'),'utc':environment()['utc']})
print(json.dumps({n:r['groups'] for n,r in results.items()},indent=2))
