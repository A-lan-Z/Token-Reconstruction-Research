"""Complete-matrix freeze, then separate evaluator-only truth scoring."""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from scripts.agent3_shortlists.core import BUDGETS, binding, metrics, paired, rank_truth, verify, write_json

DOMAINS=('pile','finance')
STAGES=(0,64,128,256)


def freeze(receipt_paths, output, source_order_path='experiments/agent3-b1-small-budget-a2/source-order-binding.json'):
    if len(receipt_paths)!=8:
        raise ValueError('all eight domain/stage cells required before truth')
    source_order=json.loads(Path(source_order_path).read_text())['order_first32_by_domain']
    cells={}
    ids={}
    common_identity=None
    for path in receipt_paths:
        receipt=json.loads(Path(path).read_text())
        key=(receipt['domain'],receipt['stage'])
        if key in cells or key[0] not in DOMAINS or key[1] not in STAGES or receipt['status']!='FROZEN_NO_TRUTH':
            raise ValueError('duplicate, unexpected, or incomplete cell')
        if receipt['truth_opened'] or receipt['target_weights_loaded']:
            raise ValueError('invalid reconstruction access')
        for b in receipt['code_files']:verify(b)
        for asset_key in ('package_manifest','lens','reference'):verify(receipt[asset_key])
        identity={key:receipt[key] for key in ('code_commit','code_files','state_sha256','readout_sha256','package_files_sha256','environment')}
        identity.update({key:receipt[key]['sha256'] for key in ('package_manifest','lens','reference')})
        if common_identity is not None and common_identity!=identity:raise ValueError('method/numerical identity differs across cells')
        common_identity=identity
        contract_path=verify(receipt['contract'])
        contract=json.loads(contract_path.read_text())
        verify(contract['observations'])
        if (contract['domain'],contract['stage'])!=key or contract['record_ids']!=receipt['record_ids']:
            raise ValueError('contract/receipt identity mismatch')
        order=receipt['record_ids']
        if order!=[r['record_id'] for r in source_order[key[0]]]:raise ValueError('source order differs from prospective ledger')
        if len(order)!=32 or len(set(order))!=32:
            raise ValueError('exact32 distinct sources/domain required')
        if key[0] in ids and ids[key[0]]!=order:
            raise ValueError('unpaired source order across stages')
        ids[key[0]]=order
        methods={m['method']:m for m in receipt['methods']}
        if set(methods)!= {'a1','b1'} or len(receipt['methods'])!=2:
            raise ValueError('both proposers required')
        for method,m in methods.items():
            tensors=load_file(verify(m['artifact']))
            if set(tensors)!= {'candidates','scores','predictions'}:
                raise ValueError('candidate archive contract changed')
            c,s,p=(tensors[k] for k in ('candidates','scores','predictions'))
            if c.shape!=(32,127,256) or s.shape!=c.shape or p.shape!=(32,128):
                raise ValueError('incomplete prediction geometry')
            if c.dtype!=torch.int64 or s.dtype!=torch.float32 or p.dtype!=torch.int64 or not torch.isfinite(s).all():
                raise ValueError('prediction dtype/finiteness')
            if (c<0).any() or (c>=128256).any() or not (p[:,0]==128000).all() or not torch.equal(p[:,1:],c[:,:,0]):
                raise ValueError('invalid prediction IDs')
            if (c.sort(-1).values.diff(dim=-1)==0).any() or (s.diff(dim=-1)>0).any():
                raise ValueError('candidate duplicate/order')
            tied=s.diff(dim=-1)==0
            if ((c.diff(dim=-1)<=0)&tied).any():
                raise ValueError('tie rule mismatch')
        cells[key]={'receipt':binding(path),'domain':key[0],'stage':key[1]}
    expected={(d,s) for d in DOMAINS for s in STAGES}
    if set(cells)!=expected:
        raise ValueError('matrix incomplete')
    result={'schema':'agent3-complete-matrix-freeze-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
            'source_order':binding(source_order_path),'cells':[cells[k] for k in sorted(cells)],'record_ids':ids,'truth_opened':False,
            'common_identity':common_identity,'scope':'paired64 sources,8 conditions,2 proposers; component diagnostic, canonical matrix incomplete'}
    write_json(output,result)
    return result


def load_frozen(freeze_path):
    f=json.loads(Path(freeze_path).read_text())
    if f['schema']!='agent3-complete-matrix-freeze-v1' or f['truth_opened']:
        raise ValueError('invalid freeze')
    # Re-validate all cells/artifacts BEFORE even opening a truth manifest.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rebuilt=freeze([verify(c['receipt']) for c in f['cells']],Path(td)/'check.json',verify(f['source_order']))
    if rebuilt['record_ids']!=f['record_ids']:
        raise ValueError('freeze identity changed')
    return f


def score(freeze_path, truth_manifest, output):
    frozen=load_frozen(freeze_path)
    truth_meta=json.loads(Path(truth_manifest).read_text())
    if set(truth_meta['domains'])!=set(DOMAINS):
        raise ValueError('truth domains mismatch')
    order=json.loads(verify(frozen['source_order']).read_text())['order_first32_by_domain']
    truth={}
    for domain in DOMAINS:
        d=truth_meta['domains'][domain]
        if d['record_ids']!=frozen['record_ids'][domain]:
            raise ValueError('evaluator source order mismatch')
        t=load_file(verify(d['artifact']))
        if set(t)!={'token_ids'} or t['token_ids'].shape!=(32,128) or t['token_ids'].dtype!=torch.int64:
            raise ValueError('truth tensor contract')
        labels=t['token_ids']
        if not (labels[:,0]==128000).all() or (labels<0).any() or (labels>=128256).any():
            raise ValueError('invalid truth IDs')
        for i,row in enumerate(order[domain]):
            actual=hashlib.sha256(labels[i].numpy().astype('<i4').tobytes()).hexdigest()
            if actual!=row['h128_sequence_sha256']:raise ValueError('truth canonical128 hash differs from frozen source ledger')
        truth[domain]=labels[:,1:]
    ranks={}
    result={'schema':'agent3-shortlist-results-v1','freeze':binding(freeze_path),'truth_manifest':binding(truth_manifest),
            'scored_utc':datetime.now(timezone.utc).isoformat(),'truth_opened_after_complete_freeze':True,'cells':[], 'contrasts':[],
            'uncertainty_unit':'source record; repeated target snapshots are paired, not independent sources',
            'rank_censoring':'257 means rank>256, not exact rank257','canonical_comparison_complete':False}
    for cell in frozen['cells']:
        receipt=json.loads(verify(cell['receipt']).read_text())
        for method in receipt['methods']:
            d,s,m=cell['domain'],cell['stage'],method['method']
            r=rank_truth(load_file(verify(method['artifact']))['candidates'],truth[d])
            ranks[d,s,m]=r
            result['cells'].append({'domain':d,'stage':s,'method':m,'ranks_through256_else257':r.tolist(),
                                    'budgets':[metrics(r,k) for k in BUDGETS]})
    decisions=[]
    for d in DOMAINS:
        for s in STAGES:
            for m in ('a1','b1'):
                for k in BUDGETS:
                    a=ranks[d,s,m]<=k
                    for ref_stage in (0,64):
                        result['contrasts'].append({'domain':d,'stage':s,'method':m,'k':k,'reference':f'{m}_stage{ref_stage}_k{k}',
                                                    **paired(a,ranks[d,ref_stage,m]<=k)})
            for k in BUDGETS:
                result['contrasts'].append({'domain':d,'stage':s,'method':'b1','k':k,'reference':f'a1_stage{s}_k{k}',
                                            **paired(ranks[d,s,'b1']<=k,ranks[d,s,'a1']<=k)})
            viable=[]
            for k in (8,16,32):
                checks=[]
                for ref in ('a1','b1'):
                    contrast=paired(ranks[d,s,'b1']<=k,ranks[d,s,ref]<=256)
                    result['contrasts'].append({'domain':d,'stage':s,'method':'b1','k':k,'reference':f'{ref}_stage{s}_k256',**contrast})
                    checks.append(contrast['empirical_candidate_loss_limits_met'])
                if all(checks):
                    viable.append(k)
            decisions.append({'domain':d,'stage':s,'empirically_promising_budgets':viable,'smallest':min(viable) if viable else None})
    result['stage1_decision']=decisions
    result['globally_empirically_promising_budgets']=[k for k in (8,16,32) if all(k in d['empirically_promising_budgets'] for d in decisions)]
    result['equivalence_established']=False
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze'); f.add_argument('--receipts',nargs='+',required=True);f.add_argument('--output',required=True);f.add_argument('--source-order',default='experiments/agent3-b1-small-budget-a2/source-order-binding.json')
    s=sub.add_parser('score');s.add_argument('--freeze',required=True);s.add_argument('--truth-manifest',required=True);s.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.receipts,a.output,a.source_order)
    else:score(a.freeze,a.truth_manifest,a.output)
