"""Retrospective evaluator; labels opened only after whole diagnostic freeze."""
from probe import *
receipt_path=X/'development_freeze.json';frozen=json.loads(receipt_path.read_text());assert frozen['binding']==bind()
rows=json.loads((P/'outputs/TRR-0016/metadata.json').read_text());assert [e['id'] for e in frozen['entries']]==[r['id'] for r in rows]
data={}
for entry in frozen['entries']:
    p=R/entry['path'];assert sha(p)==entry['sha256'];data[entry['id']]=load_file(str(p));assert set(data[entry['id']])==set(frozen['variants'])
write(X/'development_truth_gate.json',{'freeze_sha256':sha(receipt_path),'cells':len(rows)*len(frozen['variants']),'utc':time.time(),'status':'retrospective candidate recall only; no fresh evidence'})
truth_path=P/'outputs/TRR-0016/evaluator_truth.safetensors';truth=load_file(str(truth_path))
summary=defaultdict(lambda:dict(scored=0,previous_hits=0,hits=0,rescued=0,new_misses=0,complete_lists=0));misses=[]
for row in rows:
    key=row['id'];t=truth[key][1:];base=(data[key]['frequency64'][1:]==t[:,None]).any(-1)
    for variant in frozen['variants']:
        ids=data[key][variant][1:];assert ids.shape==(len(t),512 if variant=='fragment512' else 256)
        hits=(ids==t[:,None]).any(-1);s=summary[(row['setup_id'],row['condition'],row['group'],variant)]
        for k,v in [('scored',len(t)),('previous_hits',int(base.sum())),('hits',int(hits.sum())),('rescued',int((~base&hits).sum())),('new_misses',int((base&~hits).sum())),('complete_lists',int(hits.all()))]:s[k]+=v
        for pos in (~hits).nonzero().flatten().tolist():misses.append({'id':key,'position':pos+1,'token':int(t[pos]),'variant':variant,'previous_hit':bool(base[pos])})
result={'task_id':'TRR-0018','status':'RETROSPECTIVE_CANDIDATE_ONLY_NOT_RECONSTRUCTION','summary':[dict(setup_id=s,condition=c,group=g,variant=v,**d) for (s,c,g,v),d in summary.items()],'misses':misses,'truth_sha256':sha(truth_path),'freeze_sha256':sha(receipt_path),'scorer_sha256':sha(Path(__file__))}
write(X/'development_score.json',result)
for s in result['summary']:
    if s['group']!='stress':print(s['setup_id'],s['condition'],s['variant'],s['hits'],'/',s['scored'],'rescued',s['rescued'],'new_misses',s['new_misses'],flush=True)
