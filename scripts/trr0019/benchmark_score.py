from benchmark_support import *
from collections import defaultdict
import copy,numpy as np
from transformers import AutoTokenizer

def gate(receipt,rows):
    expected={(r['id'],m) for r in rows for m in METHODS}
    actual=[(r['id'],r['method']) for r in receipt['entries']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('matrix incomplete')
    if receipt['binding']!=binding():raise ValueError('binding changed')
    outputs={}
    for e in receipt['entries']:
        if e['binding']!=receipt['binding']:raise ValueError('cell binding changed')
        outputs[e['id'],e['method']]=verify(e)
    return outputs

def main():
    torch.set_num_threads(2);rows=json.loads((X/'metadata.json').read_text());receipt=json.loads((X/'prediction_freeze.json').read_text())
    predictions=gate(receipt,rows);negative=[]
    for failure in ['missing_cell','source_changed','prediction_changed']:
        bad=copy.deepcopy(receipt)
        if failure=='missing_cell':bad['entries'].pop()
        elif failure=='source_changed':bad['binding']['prefix_sha256']='0'*64
        else:bad['entries'][0]['sha256']='0'*64
        try:gate(bad,rows)
        except ValueError:negative.append(failure)
    if len(negative)!=3:raise RuntimeError('negative gate failure')
    write(X/'truth_gate.json',{'status':'COMPLETE_MATRIX_FROZEN_BEFORE_CURRENT_RETROSPECTIVE_SCORER','cells':len(predictions),
       'negative_cases_rejected':negative,'freeze_sha256':digest(X/'prediction_freeze.json'),'utc':utc()})
    truthpath=INPUT/'evaluator_truth.safetensors';truth=n.load_file(str(truthpath))
    tokenizer=AutoTokenizer.from_pretrained(n.ASSETS/'backup',local_files_only=True)
    entries={(e['id'],e['method']):e for e in receipt['entries']};scored=[];buckets=defaultdict(list)
    for row in rows:
        target=truth[row['id']];assert len(target)==row['positions'] and int(target[0])==128000
        decoded=tokenizer.decode(target[1:].tolist(),skip_special_tokens=False)
        for method in METHODS:
            p=predictions[row['id'],method];e=entries[row['id'],method]
            ok=p['tokens'][1:]==target[1:];hits=(p['candidates'][1:]==target[1:,None]).any(-1)
            median=sorted(e['phases'],key=lambda v:v['total'])[1]
            native='a1_native' if method.startswith('a1') else 'mixed_native'
            diff=p['tokens'][1:]!=predictions[row['id'],native]['tokens'][1:]
            record={**row,'method':method,'correct':int(ok.sum()),'scored':len(ok),'exact':bool(ok.all()),
              'decoded_exact':tokenizer.decode(p['tokens'][1:].tolist(),skip_special_tokens=False)==decoded,
              'proposal_hits':int(hits.sum()),'selector_errors':int((hits&~ok).sum()),'native_token_differences':int(diff.sum()),
              'seconds':e['median_seconds'],'phase_seconds':median,'candidate_simulations':e['logical_simulations']}
            scored.append(record);buckets[row['setup_id'],row['condition'],row['group'],method].append(record)
    summaries=[]
    for (setup,condition,group,method),items in buckets.items():
        denominator=sum(v['scored'] for v in items);correct=sum(v['correct'] for v in items)
        summaries.append({'setup_id':setup,'condition':condition,'group':group,'method':method,'records':len(items),
          'correct':correct,'scored':denominator,'accuracy':correct/denominator,'exact':sum(v['exact'] for v in items),
          'decoded_exact':sum(v['decoded_exact'] for v in items),'proposal_hits':sum(v['proposal_hits'] for v in items),
          'selector_errors':sum(v['selector_errors'] for v in items),'native_token_differences':sum(v['native_token_differences'] for v in items),
          'seconds_per_record':float(np.mean([v['seconds'] for v in items])),
          'mean_phases':{phase:float(np.mean([v['phase_seconds'][phase] for v in items])) for phase in items[0]['phase_seconds']},
          'candidate_simulations':sum(v['candidate_simulations'] for v in items)})
    for setup,baseline,mixed in [(CANONICAL[0],2490,2495),(CANONICAL[1],13952,13969)]:
        for method,count in [('a1_native',baseline),('a1_graph',baseline),('mixed_native',mixed),('mixed_graph',mixed)]:
            row=next(v for v in summaries if v['setup_id']==setup and v['method']==method)
            if row['correct']!=count:raise RuntimeError('accuracy reproduction anchor failed')
    pairs=[('a1_graph','a1_native'),('mixed_graph','mixed_native'),('mixed_fast','mixed_native'),
           ('mixed_fast','a1_native'),('mixed_fast','mixed_graph'),('mixed_graph','a1_graph')]
    paired=[];rng=np.random.default_rng(1919256)
    for setup,condition,group in sorted({k[:3] for k in buckets}):
        for method,control in pairs:
            aa=sorted(buckets[setup,condition,group,method],key=lambda v:v['id'])
            bb=sorted(buckets[setup,condition,group,control],key=lambda v:v['id'])
            assert [v['id'] for v in aa]==[v['id'] for v in bb]
            at=np.array([v['seconds'] for v in aa]);bt=np.array([v['seconds'] for v in bb])
            cd=np.array([a['correct']-b['correct'] for a,b in zip(aa,bb)]);den=np.array([v['scored'] for v in aa])
            ix=rng.integers(0,len(aa),size=(10000,len(aa)))
            ratios=at[ix].sum(-1)/bt[ix].sum(-1);acc=cd[ix].sum(-1)/den[ix].sum(-1)
            paired.append({'setup_id':setup,'condition':condition,'group':group,'method':method,'control':control,
                'runtime_ratio':float(at.sum()/bt.sum()),'runtime_ratio_interval95':np.quantile(ratios,[.025,.975]).tolist(),
                'correct_delta':int(cd.sum()),'accuracy_delta_interval95':np.quantile(acc,[.025,.975]).tolist(),
                'exact_delta':sum(a['exact']-b['exact'] for a,b in zip(aa,bb))})
    score={'task_id':'TRR-0019','status':'RETROSPECTIVE_FULL1360CELL_COMPARISON','summary':summaries,'paired':paired,
           'per_record':scored,'truth_path':str(truthpath),'truth_sha256':digest(truthpath),'truth_gate_sha256':digest(X/'truth_gate.json'),
           'freeze_sha256':digest(X/'prediction_freeze.json'),'actual_recovered_prefix_used':False,
           'method_changed_after_scores':False,'source_string_recovery':'not defined for truncated token clips','finished_utc':utc()}
    write(X/'score.json',score)
    inherited=json.loads((X/'inherited/canonical_matrix.json').read_text());matrix=copy.deepcopy(inherited['matrix'])
    for setup,cells in matrix.items():
        for cell in cells.values():
            cell['carried_forward_via']={'path':'experiments/TRR-0019/inherited/canonical_matrix.json','sha256':digest(X/'inherited/canonical_matrix.json')}
            cell['timing_comparable_to_current_run']=False
        row=next(v for v in summaries if v['setup_id']==setup and v['method']=='mixed_fast')
        cells['prefix_mixed256_shared_attention']={'origin':'current numerical implementation variant, retrospective port',
          'source':'experiments/TRR-0019/score.json','source_sha256':digest(X/'score.json'),'metrics':row,
          'port_differences':['batch1','trim right padding','serialization'],'decision_rule':json.loads((X/'registry.json').read_text())['new_decision_rule']}
    reg=json.loads((X/'registry.json').read_text());expected={(v['setup_id'],v['method_id']) for v in reg['required_cells']}
    actual={(s,m) for s,cells in matrix.items() for m in cells}
    if actual!=expected or len(actual)!=62:raise RuntimeError('canonical matrix incomplete')
    write(X/'canonical_matrix.json',{'required_cells':62,'complete':True,'inherited_cells':60,'new_cells':2,'matrix':matrix,
      'current_execution_controls':[v for v in summaries if v['setup_id'] in CANONICAL],
      'inherited_timings_not_compared_to_current':True,'source_registry_sha256':digest(X/'registry.json')})
    print(json.dumps({'summary':summaries,'paired':paired},indent=2),flush=True)
if __name__=='__main__':main()
