"""Verify complete immutable predictions before any evaluator label is opened."""
from native import *
import numpy as np
from collections import Counter,defaultdict

def verify(receipt,meta,panel):
    expected={(r['id'],m) for r in meta for m in ['metric64','raw64','a1a2']}
    actual=[(r['id'],r['method']) for r in receipt['entries']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete or duplicated method matrix')
    if digest(panel/'observations.safetensors')!=receipt['observations_sha256']:raise ValueError('observation changed')
    if digest(panel/'metadata.json')!=receipt['metadata_sha256']:raise ValueError('metadata changed')
    for rel,sha in receipt['implementation_hashes'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('implementation changed')
    output={};byid={r['id']:r for r in meta}
    for row in receipt['entries']:
        path=ROOT/row['path']
        if digest(path)!=row['sha256']:raise ValueError('prediction changed')
        data=load_file(str(path));L=byid[row['id']]['positions'];K=256 if row['method']=='a1a2' else 64
        if set(data)!={'tokens','candidates','scores','mse'}:raise ValueError('wrong output fields')
        if data['tokens'].shape!=(L,) or data['candidates'].shape!=(L,K) or data['scores'].shape!=(L-1,K) or data['mse'].shape!=(L-1,K):raise ValueError('wrong geometry')
        if data['tokens'].dtype!=torch.int64 or data['candidates'].dtype!=torch.int64:raise ValueError('wrong ID type')
        if int(data['tokens'][0])!=128000 or bool(((data['tokens']<0)|(data['tokens']>=128256)).any()):raise ValueError('invalid predicted token')
        # Baseline BOS proposals use the existing sentinel; only post-BOS rows are scored.
        if bool(((data['candidates'][1:]<0)|(data['candidates'][1:]>=128256)).any()):raise ValueError('invalid candidate')
        if not torch.isfinite(data['scores']).all() or not torch.isfinite(data['mse']).all():raise ValueError('nonfinite score')
        selected=data['candidates'][1:].gather(1,data['scores'].argmax(-1,keepdim=True)).squeeze(1)
        if not torch.equal(selected,data['tokens'][1:]):raise ValueError('return differs from declared decision')
        if len(row['seconds'])!=3 or not all(x>0 for x in row['seconds']):raise ValueError('missing timing')
        output[(row['id'],row['method'])]=data
    return output

def main():
    panel=OUT/'fresh_r1';receipt_path=X/'fresh_r1_prediction_receipt.json'
    receipt=json.loads(receipt_path.read_text());meta=json.loads((panel/'metadata.json').read_text())
    if len(meta)!=48 or len({r['source_id'] for r in meta})!=24:raise ValueError('source contract')
    predictions=verify(receipt,meta,panel)
    # Verify key negative gates before opening labels; no scientific outputs are rewritten.
    import copy
    rejected=[]
    broken=copy.deepcopy(receipt);broken['entries']=broken['entries'][:-1]
    try:verify(broken,meta,panel)
    except ValueError:rejected.append('missing cell')
    broken=copy.deepcopy(receipt);broken['entries'][0]['sha256']='0'*64
    try:verify(broken,meta,panel)
    except ValueError:rejected.append('changed prediction')
    broken=copy.deepcopy(receipt);key=next(iter(broken['implementation_hashes']));broken['implementation_hashes'][key]='0'*64
    try:verify(broken,meta,panel)
    except ValueError:rejected.append('changed implementation')
    if len(rejected)!=3:raise RuntimeError('negative gate failure')
    write(X/'fresh_truth_gate.json',{'prediction_receipt_sha256':digest(receipt_path),'verified_predictions':len(predictions),'negative_cases_rejected':rejected,'truth_opened':False,'utc':environment()['utc']})
    cap=json.loads((X/'fresh_capture.json').read_text())
    if digest(panel/'evaluator_truth.json')!=cap['truth_sha256']:raise ValueError('truth binding changed')
    truth=json.loads((panel/'evaluator_truth.json').read_text())
    if set(truth)!={r['id'] for r in meta}:raise ValueError('truth IDs changed')
    allrows=[];entries={(r['id'],r['method']):r for r in receipt['entries']}
    for row in meta:
        true=torch.tensor(truth[row['id']])
        for m in ['metric64','raw64','a1a2']:
            p=predictions[(row['id'],m)];correct=p['tokens'][1:]==true[1:];hits=(p['candidates'][1:]==true[1:,None]).any(-1)
            allrows.append({**row,'method':m,'correct':int(correct.sum()),'scored':len(true)-1,'exact':bool(correct.all()),'proposal_hits':int(hits.sum()),'wrong_despite_inclusion':int((~correct&hits).sum()),'first_error':next((i+1 for i,x in enumerate(correct) if not x),None),'seconds':entries[(row['id'],m)]['median_seconds']})
    summaries=[]
    for c in ['matched','lora256']:
        for g in ['natural','stress']:
            for m in ['metric64','raw64','a1a2']:
                rr=[r for r in allrows if r['condition']==c and r['group']==g and r['method']==m]
                n=sum(r['scored'] for r in rr);right=sum(r['correct'] for r in rr)
                summaries.append({'condition':c,'group':g,'method':m,'correct':right,'scored':n,'accuracy':right/n,'exact':sum(r['exact'] for r in rr),'records':len(rr),'proposal_hits':sum(r['proposal_hits'] for r in rr),'wrong_despite_inclusion':sum(r['wrong_despite_inclusion'] for r in rr),'seconds':sum(r['seconds'] for r in rr)})
    paired=[];rng=np.random.default_rng(1414)
    for c in ['matched','lora256']:
        for g in ['natural','stress']:
            a=[r for r in allrows if r['condition']==c and r['group']==g and r['method']=='metric64']
            b=[r for r in allrows if r['condition']==c and r['group']==g and r['method']=='a1a2']
            assert [r['id'] for r in a]==[r['id'] for r in b]
            dif=np.array([(x['correct']-y['correct'])/x['scored'] for x,y in zip(a,b)])
            ex=np.array([int(x['exact'])-int(y['exact']) for x,y in zip(a,b)])
            ix=rng.integers(0,len(a),(10000,len(a)))
            paired.append({'condition':c,'group':g,'token_delta':float(dif.mean()),'token_95_bootstrap':np.quantile(dif[ix].mean(1),[.025,.975]).tolist(),'exact_delta':float(ex.mean()),'exact_95_bootstrap':np.quantile(ex[ix].mean(1),[.025,.975]).tolist(),'time_ratio':sum(x['seconds'] for x in a)/sum(x['seconds'] for x in b)})
    report={'status':'FRESH_TASK_LOCAL_PANEL_SCORED_AFTER_COMPLETE_FREEZE','scope':'exploratory direction; canonical comparison incomplete; supplied public prefix, not recovered weights','summary':summaries,'paired':paired,'per_record':allrows,'truth_gate_sha256':digest(X/'fresh_truth_gate.json'),'prediction_receipt_sha256':digest(receipt_path),'scoring_environment':environment(),'uncertainty':'source bootstrap within tiny two-book/generator panel, not population-wide or simultaneous bounds'}
    write(X/'fresh_score.json',report);print(json.dumps({'summary':summaries,'paired':paired},indent=2),flush=True)
if __name__=='__main__':main()
