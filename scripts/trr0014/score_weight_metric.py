from native import *
dst=X/'weight_metric_dev_r1'
freeze=json.loads((dst/'freeze.json').read_text())
for f in freeze['files']:
    if digest(ROOT/f['path'])!=f['sha256']:raise ValueError('modified evidence')
truth=json.loads((RESCUE/'development/evaluator_truth.json').read_text())
metrics=[]
for method in ['total','diagonal','mlp']:
    m={'method':method,'correct':0,'denominator':0,'hits':0,'exact':0,'seconds':0}
    for key,t in truth.items():
        r=json.loads((dst/(method+'_'+key+'.json')).read_text())
        m['correct']+=sum(a==b for a,b in zip(r['tokens'][1:],t[1:]));m['denominator']+=len(t)-1
        m['hits']+=sum(t[z['position']] in {v['token'] for v in z['checks']} for z in r['trace'])
        m['exact']+=int(r['tokens']==t);m['seconds']+=r['seconds']
    metrics.append(m)
write(dst/'score.json',{'metrics':metrics,'scope':'retrospective development only','freeze_sha256':digest(dst/'freeze.json')})
print(json.dumps(metrics,indent=2))
