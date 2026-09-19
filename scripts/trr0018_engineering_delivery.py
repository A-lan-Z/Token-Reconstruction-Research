"""Audit and report post-confirmation exact execution optimization."""
from pathlib import Path
import json,hashlib,subprocess,statistics,time,sys
from collections import defaultdict
import numpy as np
from safetensors.torch import load_file
import torch
from trr0018_delivery import archive,digest,write,R,X,OUT

def main():
    torch.set_num_threads(2);receipt_path=X/'engineering_receipt.json';receipt=json.loads(receipt_path.read_text())
    rows=json.loads((OUT/'metadata.json').read_text());variants=('mixed_native','mixed_shared','a1a2_native')
    expected={(r['id'],v) for r in rows for v in variants};actual=[(e['id'],e['variant']) for e in receipt['entries']]
    assert len(actual)==1296 and len(set(actual))==1296 and set(actual)==expected
    original=json.loads((X/'prediction_receipt.json').read_text());assert receipt['binding']['original']==original['binding']
    assert receipt['binding']['frozen_reference_receipt_sha256']==digest(X/'prediction_receipt.json')
    commit=receipt['setup']['environment']['commit'];checks={}
    for rel,sha in receipt['binding']['additional_hashes'].items():
        assert digest(R/rel)==sha
        assert hashlib.sha256(subprocess.check_output(['git','-C',str(R),'show',commit+':'+rel])).hexdigest()==sha
        checks[rel]=sha
    references={};groups=defaultdict(list);golden={(e['id'],e['method']):e for e in original['entries']}
    for e in receipt['entries']:
        assert e['binding']==receipt['binding'] and e['all_arrays_byte_identical']
        ref=e['reference_path']
        expected_ref=golden[(e['id'],'a1a2' if e['variant']=='a1a2_native' else 'mixed256')]
        assert (ref,e['reference_sha256'])==(expected_ref['path'],expected_ref['sha256'])
        assert e['logical_simulations']==(e['positions']-1)*256
        if ref not in references:
            p=R/ref;assert digest(p)==e['reference_sha256'];data=load_file(str(p))
            references[ref]={k:{'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()} for k,v in data.items()}
        assert e['computed_tensor_hashes']==references[ref]
        groups[(e['setup_id'],e['condition'],e['group'],e['variant'])].append(e)
    guard=json.loads((X/'engineering_guard.json').read_text());assert guard['returncode']==0 and guard['failure'] is None
    summary=[];paired=[];outliers=[];rng=np.random.default_rng(1818257)
    for (setup,condition,group,variant),entries in groups.items():
        seconds=[e['seconds'] for e in entries];med=statistics.median(seconds)
        summary.append({'setup_id':setup,'condition':condition,'group':group,'variant':variant,'records':len(entries),
                        'total_seconds':sum(seconds),'mean_record_seconds':statistics.mean(seconds),
                        'phase_seconds':{k:sum(e['phases'][k] for e in entries) for k in entries[0]['phases']},
                        'logical_simulations':sum(e['logical_simulations'] for e in entries)})
        for e in entries:
            if e['seconds']>2*med:outliers.append({'id':e['id'],'variant':variant,'seconds':e['seconds'],'disposition':'retained; no rerun or exclusion'})
    for setup,condition,group in sorted({k[:3] for k in groups}):
        a=groups[(setup,condition,group,'mixed_shared')]
        for comparator in ('mixed_native','a1a2_native'):
            bm={e['id']:e for e in groups[(setup,condition,group,comparator)]};b=[bm[e['id']] for e in a]
            aa=np.array([e['seconds'] for e in a]);bb=np.array([e['seconds'] for e in b]);indices=rng.integers(0,len(a),(10000,len(a)))
            paired.append({'setup_id':setup,'condition':condition,'group':group,'comparator':comparator,'time_ratio':float(aa.mean()/bb.mean()),
                           'paired_record_bootstrap95':np.quantile(aa[indices].sum(1)/bb[indices].sum(1),[.025,.975]).tolist(),
                           'median_paired_ratio_posthoc':float(np.median(aa/bb))})
    score={'task_id':'TRR-0018','status':'RETROSPECTIVE_EXACT_EXECUTION_EQUIVALENCE_AND_PAIRED_TIMING','summary':summary,'paired':paired,'outliers':outliers,
           'timed_invocations_per_cell':1,'qualification_repetitions':3,'bootstrap_seed':1818257,'bootstrap_draws':10000,
           'all_1296_cells_match_frozen_reference':True,'shared_inputs':432,'new_accuracy_claim':False,
           'receipt_sha256':digest(receipt_path),'execution_commit':commit,'additional_hashes':checks,
           'prediction_wall_seconds':guard['end_unix']-guard['start_unix'],'peak_allocated':receipt['peak_allocated'],'peak_reserved':receipt['peak_reserved'],
           'memory_scope':'whole comparison process with all prepared assets','failure':guard['failure'],
           'source_optimization':'Exact TRR-0017 shared_context.py adapter, copied without changes; qualified again on current candidate lists',
           'runtime_claim_boundary':'same process, rotating order, exclusive CUDA; one timed invocation per method/input; intervals across paired records, not repeated latency distributions'}
    write(X/'engineering_score.json',score)
    new_archive=archive('engineering-receipts',sorted((OUT/'engineering_receipts').glob('*.json')))
    index_path=X/'artifact_index.json';index=json.loads(index_path.read_text());index['archives'].append(new_archive);index['count']=len(index['archives']);index['bytes']=sum(a['bytes'] for a in index['archives']);index['member_count']=sum(len(a['members']) for a in index['archives']);index['engineering_arrays']='Independently produced arrays byte-identical to original prediction archives; computed tensor hashes and reference hashes in engineering receipts'
    index_path.write_text(json.dumps(index,indent=2)+'\n')
    audit={'task_id':'TRR-0018','engineering_execution_commit':commit,'new_files_match_exact_commit':checks,'current_cells':1296,'distinct_reference_arrays':len(references),
           'all_computed_tensor_hashes_match_archived_references':True,'engineering_receipt_sha256':digest(receipt_path),'engineering_archive_sha256':new_archive['sha256'],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(X/'engineering_audit.json',audit)
    report=R/'coordination/results/TRR-0018.md';lines=['','## Combined method: exact shared-context A2','',
    'The tokenizer/lookup decision rule stayed frozen after R4 scoring. A separate engineering study then applied the existing TRR-0017 cache adapter: it references the committed context, expands read-only per layer and materializes the exact native candidate tensors. This removes copying while retaining every candidate simulation. Cache sharing is a general A2 optimization that can also be applied to A1+A2. This timing study retains the existing native A1+A2 implementation as its control; it does not measure an equally cache-optimized A1+A2 control or establish that the new proposer has lower intrinsic overhead. The128-position qualification compared every candidate hidden value at all127 scored positions, verified cache immutability, and repeated complete outputs three times. The first qualification harness attempt inspected unpopulated cache layers; that failure and the correction are preserved in attempts.json. The adapter itself did not change.','',
    'Across all432 inputs, both native controls and shared execution exactly reproduce every archived token, ordered candidate, cosine score and MSE value. The1,296 engineering timing cells each contain one timed invocation. All raw times are retained. These timing measurements occurred after truth opening; accuracy is inherited through exact output equivalence to the prior complete freeze, not presented as another fresh confirmation.','',
    '| Panel | Shared mixed seconds | Native mixed seconds | Native A1+A2 seconds | Shared/native mixed ratio | Shared/baseline ratio |','|---|---:|---:|---:|---:|---:|']
    sm={(v['setup_id'],v['condition'],v['group'],v['variant']):v for v in summary}
    for setup,condition,group in sorted({k[:3] for k in sm}):
        a=sm[(setup,condition,group,'mixed_shared')]['mean_record_seconds'];b=sm[(setup,condition,group,'mixed_native')]['mean_record_seconds'];c=sm[(setup,condition,group,'a1a2_native')]['mean_record_seconds']
        lines.append(f'| {setup} / {condition} / {group} | {a:.5f} | {b:.5f} | {c:.5f} | {a/b:.4f} | {a/c:.4f} |')
    lines += ['', 'The paired bootstrap timing intervals, per-record receipts, any retained outliers and the execution binding are in engineering_score.json and engineering_receipt.json. The engineering matrix covers both canonical setups and every auxiliary condition. This is a numerically equivalent execution variant of the registered mixed method, so the active canonical decision-rule matrix remains60 cells. Current memory peaks hold all comparison assets and are not standalone per-method memory measurements.','',
    'Fresh R4 prose has4 errors out of4064 scored tokens in each target condition, compared with6/4064 for the previous voting rule and3/4064 matched or2/4064 shifted for A1+A2. Exact clips improve from26/32 to28/32 in each condition; baseline reaches29/32 and30/32. Thus the accuracy improvement is modest and the baseline still has a small fresh advantage. The original R3 panel now matches baseline token and exact-clip scores; Pile is unchanged from voting, while Finance gains two correct tokens and two exact rows. No general equivalence or baseline-replacement claim is made.','',
    'The combined proposer plus shared-cache execution remains prefix-only in the intended sense: no separately fitted token predictor, no extra learned state, and exactly256 A2 simulations per scored token. It still uses the supplied public prefix; changing recovered-prefix trajectories remain future work. The research goal is unfinished.']
    report.write_text(report.read_text()+'\n'.join(lines)+'\n')
    matrix_path=X/'canonical_matrix.json';matrix=json.loads(matrix_path.read_text());matrix['output_equivalent_execution_study']={'source':'experiments/TRR-0018/engineering_score.json','sha256':digest(X/'engineering_score.json'),'decision_rule_unchanged':True,'canonical_summary':[v for v in summary if v['setup_id'].startswith(('clean-','historical-'))],'all_432_shared_arrays_equal':True,'timing_invocations_per_cell':1};matrix_path.write_text(json.dumps(matrix,indent=2)+'\n')
    manifest_path=X/'manifest.json';manifest=json.loads(manifest_path.read_text());manifest['engineering']=score;manifest['artifact_count']=index['count'];manifest['artifact_bytes']=index['bytes'];manifest['result_sha256']=digest(report);manifest['engineering_delivery_script_sha256']=digest(Path(__file__));manifest['receipts']={str(p.relative_to(R)):digest(p) for p in sorted(X.glob('*.json')) if p.name!='manifest.json'};manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    state=json.loads((R/'coordination/STATE.json').read_text());state.update(status='ACCURACY_AND_EXACT_EXECUTION_OPTIMIZATION_VERIFIED_GOAL_OPEN',engineering_execution_commit=commit,shared_cache_equivalence_inputs=432,goal_completed=False);(R/'coordination/STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'summary':summary,'paired':paired,'outliers':outliers},indent=2),flush=True)
if __name__=='__main__':main()
