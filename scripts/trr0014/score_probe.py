from native import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--name',default='lookup_dev_r1');args=p.parse_args()
dst=X/args.name
freeze=json.loads((dst/'freeze.json').read_text())
for f in freeze['files']:
    if digest(ROOT/f['path'])!=f['sha256']:raise ValueError('prediction changed')
# Open previously scored development truth only after the complete prediction freeze.
truth=json.loads((RESCUE/'development/evaluator_truth.json').read_text())
print('truth structure',type(truth).__name__,flush=True)
if isinstance(truth,list):truth={r['id']:r['tokens'] for r in truth}
metrics=[]
for method in ['euclidean','cosine','transport','a1a2']:
    out={'method':method,'correct':0,'tokens':0,'exact':0,'records':0,'proposal_hits':0,'seconds':0,'details':[]}
    for key,t in truth.items():
        if isinstance(t,dict):t=t.get('tokens',t.get('input_ids'))
        r=json.loads((dst/(method+'_'+key+'.json')).read_text())
        correct=sum(a==b for a,b in zip(r['tokens'][1:],t[1:]))
        hits=sum(t[z['position']] in {q['token'] for q in z['checks']} for z in r['trace'])
        out['correct']+=correct;out['tokens']+=len(t)-1;out['exact']+=int(r['tokens']==t);out['records']+=1;out['proposal_hits']+=hits;out['seconds']+=r['seconds']
        out['details'].append({'record':key,'correct':correct,'denominator':len(t)-1,'hits':hits,'first_error':next((i for i in range(1,len(t)) if r['tokens'][i]!=t[i]),None)})
    metrics.append(out)
write(dst/'score.json',{'freeze_sha256':digest(dst/'freeze.json'),'metrics':metrics,'scope':'retrospective development only'})
print(json.dumps(metrics,indent=2),flush=True)
