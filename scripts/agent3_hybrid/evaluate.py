"""Full-matrix freeze and evaluator-only source-paired static-hybrid scoring."""
from __future__ import annotations
import argparse,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from scripts.agent3_hybrid.common import *
from scripts.agent3_shortlists.core import bootstrap_mean

def freeze(receipts,panel_path,output):
    check_implementation();panel=json.loads(Path(panel_path).read_text());cells={};identity=None
    if len(receipts)!=8:raise ValueError('eight cells required before truth')
    for path in receipts:
        c=json.loads(Path(path).read_text());key=(c['domain'],c['stage'])
        if key in cells or c['status']!='FROZEN_CELL_NO_TRUTH' or c['source_truth_loaded'] or c['target_weights_loaded']:raise ValueError('invalid or duplicate cell')
        for b in c['package_modules'].values():verify(b)
        ident=(c['code_commit'],c['implementation']['sha256'],c['state_sha256'],c['readout_sha256'])
        if identity is not None and identity!=ident:raise ValueError('execution identity differs')
        identity=ident;verify(c['implementation']);contract=json.loads(verify(c['contract']).read_text());verify(contract['observations'])
        if c['record_ids']!=[r['record_id'] for r in panel['records'][key[0]]] or contract['record_ids']!=c['record_ids']:raise ValueError('source pairing mismatch')
        if set(m['method'] for m in c['methods'])!=set(METHODS) or len(c['methods'])!=5:raise ValueError('five fixed arms required')
        for m in c['methods']:
            t=load_file(verify(m['artifact']));k=1 if m['method']=='b1_alone' else int(m['method'].split('_k')[1])
            if set(t)!={'predictions','candidates','proposal_scores','a2_scores','committed_tokens'}:raise ValueError('artifact fields changed')
            if t['predictions'].shape!=(32,128) or t['candidates'].shape!=(32,127,k) or t['proposal_scores'].shape!=t['candidates'].shape:raise ValueError('incomplete artifact')
            if not (t['predictions'][:,0]==128000).all() or not t['candidates'].eq(t['predictions'][:,1:,None]).any(-1).all():raise ValueError('output outside fixed lists')
            if not torch.isfinite(t['proposal_scores']).all() or not torch.isfinite(t['a2_scores']).all():raise ValueError('nonfinite retained scores')
            if (t['candidates'].sort(-1).values.diff(dim=-1)==0).any():raise ValueError('duplicate proposals')
            if k>1:
                if t['a2_scores'].shape!=(32,127,k) or not torch.equal(t['committed_tokens'],t['predictions']):raise ValueError('cache trace mismatch')
                selected=t['candidates'].gather(-1,t['a2_scores'].argmax(-1,keepdim=True)).squeeze(-1)
                if not torch.equal(selected,t['predictions'][:,1:]):raise ValueError('native direct-cosine winner mismatch')
            if len(m['timing_records'])!=32 or any(not x['repeated_prediction_exact'] for x in m['timing_records']):raise ValueError('repeat failure')
        cells[key]={'domain':key[0],'stage':key[1],'receipt':binding(path)}
    if set(cells)!={(d,s) for d in DOMAINS for s in STAGES}:raise ValueError('incomplete matrix')
    result={'schema':'agent3-static-hybrid-complete-freeze-v1','status':'ALL40_OUTPUTS_FROZEN_BEFORE_TRUTH','frozen_utc':utc(),'panel':binding(panel_path),'implementation':binding(EV/'implementation-freeze.json'),'cells':[cells[k] for k in sorted(cells)],'source_truth_opened':False,'code_identity':list(identity)}
    write_json(output,result);return result

def load_freeze(path):
    f=json.loads(Path(path).read_text())
    if f['status']!='ALL40_OUTPUTS_FROZEN_BEFORE_TRUTH' or f['source_truth_opened']:raise ValueError('invalid complete freeze')
    with tempfile.TemporaryDirectory() as d:
        rebuilt=freeze([verify(c['receipt']) for c in f['cells']],verify(f['panel']),Path(d)/'validated.json')
    if rebuilt['code_identity']!=f['code_identity']:raise ValueError('freeze identity changed')
    return f

def error_counts(predictions,candidates,truth):
    pred=np.asarray(predictions);y=np.asarray(truth);ids=np.asarray(candidates)
    if pred.shape!=y.shape or ids.shape[:2]!=(pred.shape[0],pred.shape[1]-1):raise ValueError('geometry mismatch')
    correct=pred[:,1:]==y[:,1:];included=(ids==y[:,1:,None]).any(-1);wrong=~correct
    earlier=np.cumsum(wrong,axis=1)-wrong>0
    counts={'scored_tokens':int(correct.size),'correct_tokens':int(correct.sum()),'token_accuracy':bootstrap_mean(correct.mean(1),seed=9103),'exact_clips':int(correct.all(1).sum()),'exact_clip_recovery':bootstrap_mean(correct.all(1),seed=9103),'shortlist_omissions':int((~included).sum()),'shortlist_recall':bootstrap_mean(included.mean(1),seed=9103),'wrong_despite_inclusion':int((wrong&included).sum()),'wrong_with_omission':int((wrong&~included).sum()),'wrong_after_earlier_wrong_commitment':int((wrong&earlier).sum()),'included_but_wrong_after_earlier_wrong':int((wrong&included&earlier).sum()),'omitted_and_wrong_after_earlier_wrong':int((wrong&~included&earlier).sum()),'included_but_wrong_before_any_wrong':int((wrong&included&~earlier).sum()),'omitted_and_wrong_before_any_wrong':int((wrong&~included&~earlier).sum()),'correct_after_earlier_wrong':int((correct&earlier).sum()),'first_error_positions':[int(np.flatnonzero(row)[0]+1) if row.any() else None for row in wrong],'errors_per_record':wrong.sum(1).tolist(),'omissions_per_record':(~included).sum(1).tolist(),'conditional_selection_accuracy':None if not included.any() else float((correct&included).sum()/included.sum()),'association_not_proof_of_causal_propagation':True}
    return counts,correct,included

def contrast(a,b,ta,tb):
    from scipy.stats import beta
    n=len(a);lost=(b&~a).any(1);x=int(lost.sum())
    token=bootstrap_mean(a.mean(1)-b.mean(1),seed=9103);clip=bootstrap_mean(a.all(1).astype(float)-b.all(1).astype(float),seed=9103)
    rng=np.random.default_rng(9113);ix=rng.integers(n,size=(10000,n));ratios=ta[ix].sum(1)/tb[ix].sum(1);ratio=float(ta.sum()/tb.sum())
    return {'token_delta':token,'exact_clip_delta':clip,'records_with_any_lost_correct_token':x,'record_loss_probability_one_sided95_upper':1. if x==n else float(beta.ppf(.95,x+1,n-x)),'warmed_time_ratio':ratio,'warmed_ratio_paired_source_bootstrap95':np.quantile(ratios,[.025,.975]).tolist(),'empirical_quality_limits_met':token['estimate']>=-.001-1e-12 and clip['estimate']>=-.02-1e-12,'empirical_acceleration_met':ratio<=.8,'equivalence_established':False}

def score(a):
    f=load_freeze(a.freeze);panel=json.loads(verify(f['panel']).read_text());truth_manifest=json.loads(Path(a.truth).read_text());truth={}
    for d in DOMAINS:
        meta=truth_manifest['domains'][d];labels=load_file(verify(meta['artifact']))['token_ids']
        if labels.shape!=(32,128) or meta['record_ids']!=[r['record_id'] for r in panel['records'][d]]:raise ValueError('truth order/geometry mismatch')
        for i,row in enumerate(panel['records'][d]):
            if hashlib.sha256(labels[i].numpy().astype('<i4').tobytes()).hexdigest()!=row['h128_sequence_sha256']:raise ValueError('truth canonical identity mismatch')
        truth[d]=labels.numpy()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6',local_files_only=True)
    result={'status':'SCORED_AFTER_COMPLETE_FREEZE','freeze':binding(a.freeze),'truth':binding(a.truth),'scored_utc':utc(),'code_commit':commit(),'cells':[],'contrasts':[],'static_public_prefix':True,'active_prefix_recovery_tested':False,'canonical_comparison_complete':False,'population_equivalence_established':False};arrays={}
    for cell in f['cells']:
        receipt=json.loads(verify(cell['receipt']).read_text());d,s=cell['domain'],cell['stage']
        for m in receipt['methods']:
            t=load_file(verify(m['artifact']));counts,correct,included=error_counts(t['predictions'].numpy(),t['candidates'].numpy(),truth[d]);timings=np.array([np.median(x['measured_seconds']) for x in m['timing_records']])
            decoded=[tokenizer.decode(pred.tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)==tokenizer.decode(y.tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False) for pred,y in zip(t['predictions'],truth[d],strict=True)]
            counts['decoded_token_text_exact_clips']=sum(decoded)
            counts['original_source_string_exactness']='not scored; fixed128 token clip may truncate source string'
            counts.update(domain=d,stage=s,method=m['method'],warmed_median_seconds_per_record=timings.tolist(),warmed_total_seconds=float(timings.sum()),peak_resources=m['peak_resources'],phase_records=m['phase_records'],selector_present=m['method']!='b1_alone')
            result['cells'].append(counts);arrays[d,s,m['method']]=(correct,timings)
        for method in METHODS:
            for reference in ('a1_k256','b1_k256','b1_alone'):
                x,tx=arrays[d,s,method];y,ty=arrays[d,s,reference]
                result['contrasts'].append({'domain':d,'stage':s,'method':method,'reference':reference,**contrast(x,y,tx,ty)})
    decisions=[]
    for d in DOMAINS:
        for s in STAGES:
            refs=[c for c in result['contrasts'] if c['domain']==d and c['stage']==s and c['method']=='b1_k16' and c['reference'] in ('a1_k256','b1_k256')]
            decisions.append({'domain':d,'stage':s,'quality_met':all(c['empirical_quality_limits_met'] for c in refs),'acceleration_met':all(c['empirical_acceleration_met'] for c in refs)})
    result['decision_cells']=decisions;result['retain_static_prefix_candidate']=all(c['quality_met'] and c['acceleration_met'] for c in decisions)
    write_json(a.output,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='phase',required=True)
    f=sub.add_parser('freeze');f.add_argument('--receipts',nargs='+',required=True);f.add_argument('--panel',required=True);f.add_argument('--output',required=True)
    s=sub.add_parser('score');s.add_argument('--freeze',required=True);s.add_argument('--truth',required=True);s.add_argument('--output',required=True)
    a=p.parse_args()
    if a.phase=='freeze':freeze(a.receipts,a.panel,a.output)
    else:score(a)
