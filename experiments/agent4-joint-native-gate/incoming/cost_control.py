import torch,json,time,hashlib
from pathlib import Path
from prototype import *
R=Path(__file__).parent

def main():
 torch.set_num_threads(1); rows=[]
 for seed in [200,201,202]:
  m=TinyPrefix(seed); g=torch.Generator().manual_seed(seed+10000)
  for r in range(16):
   truth=torch.randint(1,128,(8,),generator=g); seq=torch.cat((torch.tensor([0]),truth))
   with torch.no_grad(): obs=m.forward(m.E[seq][None])[0,1:]
   out=window_search(m,obs,'sequential',p=16,fast=True,block_commit=True)
   out.update(seed=seed,record=r,method='sequential_p16',truth=truth.tolist())
   out['correct']=sum(x==y for x,y in zip(out['prediction'],truth.tolist()));out['exact']=out['correct']==8
   rows.append(out)
 summary={'correct':sum(x['correct'] for x in rows),'tokens':384,'exact':sum(x['exact'] for x in rows),'sequences':48,
          'seconds':sum(x['seconds'] for x in rows),'work':{k:sum(x['work'][k] for x in rows) for k in rows[0]['work']}}
 out={'scope':'Retrospective single wider-search cost control, same second cohort. Not tuned over a grid.','summary':summary,'rows':rows}
 (R/'results_cost_control.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
