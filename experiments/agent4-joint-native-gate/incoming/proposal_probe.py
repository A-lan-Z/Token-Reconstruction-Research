"""Post-score mechanistic ablation using opened second-cohort observations.
Truth is NOT passed to proposal generation. Correctness categories are assigned
only after candidates have been constructed. This is not a fresh benchmark.
"""
import torch,json
from pathlib import Path
from prototype import *
R=Path(__file__).parent

def main():
 torch.set_num_threads(1)
 data=json.loads((R/'results_v2.json').read_text())
 baselines={(x['seed'],x['record']):x for x in data['rows'] if x['method']=='sequential'}
 probes=[]
 for seed in [200,201,202]:
  m=TinyPrefix(seed); g=torch.Generator().manual_seed(seed+10000)
  for r in range(16):
   truth=torch.randint(1,128,(8,),generator=g); seq=torch.cat((torch.tensor([0]),truth))
   with torch.no_grad(): obs=m.forward(m.E[seq][None])[0,1:].clone()
   pred=baselines[seed,r]['prediction']; cache=m.commit(torch.tensor([0]),m.empty_cache())
   for i in range(8):
    w=min(3,8-i)
    with torch.no_grad(): init=stable_nearest(m.E,obs[i:i+w],1).squeeze(-1)
    z=m.E[init][None].detach().requires_grad_(True)
    y=m.forward(z,cache); l=observed_losses(y,obs[i:i+w])
    gl=torch.autograd.grad(l[:,0].sum(),z,retain_graph=True)[0][:,:1]
    gj=torch.autograd.grad(l.sum(),z)[0][:,:1]
    with torch.no_grad():
     local=proposals(m.E,z[:,:1],gl,2)[0,0].tolist()
     joint=proposals(m.E,z[:,:1],gj,2)[0,0].tolist()
    # All target comparisons occur after the above observable-only proposal step.
    prior_ok=pred[:i]==truth[:i].tolist()
    probes.append({'seed':seed,'record':r,'position':i,'width':w,'local':local,'joint':joint,
                   'truth':int(truth[i]),'correct_prefix':prior_ok,
                   'baseline_root':prior_ok and pred[i]!=int(truth[i])})
    cache=m.commit(torch.tensor([pred[i]]),cache)
 subsets={}
 for label,sel in [('all',lambda x:True),('correct_prefix',lambda x:x['correct_prefix']),('baseline_roots',lambda x:x['baseline_root'])]:
  rr=[x for x in probes if sel(x)]
  subsets[label]={'positions':len(rr),'local_top2_hits':sum(x['truth'] in x['local'] for x in rr),
                 'joint_top2_hits':sum(x['truth'] in x['joint'] for x in rr),
                 'joint_only':sum(x['truth'] in x['joint'] and x['truth'] not in x['local'] for x in rr),
                 'local_only':sum(x['truth'] in x['local'] and x['truth'] not in x['joint'] for x in rr)}
 out={'scope':'Retrospective proposal-only ablation, same provisional blocks and baseline-produced prefix per position; not Llama or independent confirmation.','subsets':subsets,'rows':probes}
 (R/'proposal_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(subsets,indent=2))
if __name__=='__main__':main()
