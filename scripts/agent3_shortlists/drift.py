"""Paired boundary-activation displacement from permitted observations only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from scripts.agent3_shortlists.core import binding, verify, write_json


def measure(contracts, output):
    torch.set_num_threads(2)
    cells={};records={};bindings=[]
    for path in contracts:
        c=json.loads(Path(path).read_text());key=(c['domain'],c['stage'])
        if key in cells:raise ValueError('duplicate observation condition')
        if c['domain'] in records and records[c['domain']]!=c['record_ids']:raise ValueError('source order is not paired')
        records[c['domain']]=c['record_ids']
        t=load_file(verify(c['observations']))
        if set(t)!={'activations','attention_mask','position_ids'}:raise ValueError('forbidden observation fields')
        if not (t['attention_mask']==1).all() or t['activations'].shape[1:]!=(128,2048):raise ValueError('fixed128 geometry required')
        if not torch.equal(t['position_ids'],torch.arange(128).expand(len(c['record_ids']),-1)):raise ValueError('positions changed')
        cells[key]=t['activations'][:,1:].double()
        bindings.append(binding(path))
    expected={(d,s) for d in ('pile','finance') for s in (0,64,128,256)}
    if set(cells)!=expected:raise ValueError('requires complete paired observation matrix')
    result={'schema':'agent3-boundary-drift-v1','observation_contracts':bindings,'truth_opened':False,'scored_positions':'1..127','comparisons':[]}
    for d,s in sorted(cells):
        for ref in (0,64):
            h,b=cells[d,s],cells[d,ref]
            delta=h-b
            relative=delta.flatten(1).norm(dim=1)/b.flatten(1).norm(dim=1)
            cosine=torch.nn.functional.cosine_similarity(h,b,dim=-1)
            result['comparisons'].append({'domain':d,'stage':s,'reference_stage':ref,
                  'relative_l2_per_record':relative.tolist(),'relative_l2_mean':relative.mean().item(),
                  'cosine_distance_per_record':(1-cosine).mean(1).tolist(),'cosine_distance_mean':(1-cosine).mean().item(),
                  'changed_elements':int(delta.ne(0).sum()),'total_elements':delta.numel(),
                  'postbos_position_cosine_distance_quantiles':torch.quantile((1-cosine).flatten(),torch.tensor([0.,.5,.9,.99,1.],dtype=torch.float64)).tolist()})
    write_json(output,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contracts',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();measure(a.contracts,a.output)
