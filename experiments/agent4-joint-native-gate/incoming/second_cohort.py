"""Prospectively defined second synthetic cohort; no fitted model or natural text."""
import json,time,hashlib,platform
from pathlib import Path
import torch
from prototype import TinyPrefix,window_search,DTYPE
from run_experiments_v1 import analytic_checks,cache_checks
ROOT=Path(__file__).parent

def main():
 torch.set_num_threads(1)
 rows=[]; methods=['sequential','diagonal','joint']
 for seed in [200,201,202]:
  m=TinyPrefix(seed); g=torch.Generator().manual_seed(seed+10000)
  for r in range(16):
   truth=torch.randint(1,128,(8,),generator=g)
   seq=torch.cat((torch.tensor([0]),truth))
   with torch.no_grad(): obs=m.forward(m.E[seq][None])[0,1:]
   for mode in methods[r%3:]+methods[:r%3]:
    out=window_search(m,obs,mode,fast=True,block_commit=True)
    out.update(seed=seed,record=r,method=mode,truth=truth.tolist())
    out['correct']=sum(a==b for a,b in zip(out['prediction'],truth.tolist()))
    out['exact']=out['correct']==8; rows.append(out)
 summary={}
 for mode in methods:
  rr=[r for r in rows if r['method']==mode]
  summary[mode]={'correct':sum(r['correct'] for r in rr),'tokens':384,'exact':sum(r['exact'] for r in rr),'sequences':48,
    'cpu_seconds':sum(r['seconds'] for r in rr),'work':{k:sum(r['work'][k] for r in rr) for k in rr[0]['work']}}
 out={'scope':'Independent second synthetic random-model cohort, not Llama. No fitting.',
      'config':{'seeds':[200,201,202],'records_per_seed':16,'length':8,'window':3,'beam':2,'proposals':2,'rounds':4,'fast':True,'block_commit':True,'match_sum_loss_threshold':1e-20},
      'checks':{'analytic':analytic_checks(),'cache':cache_checks()},'summary':summary,'rows':rows}
 for p in ['PLAN.md','AMENDMENT.md','prototype.py',Path(__file__).name]:
  out.setdefault('source_sha256',{})[p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
 out['environment']={'torch':torch.__version__,'threads':1,'cpu_only':True}
 (ROOT/'results_v2.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
