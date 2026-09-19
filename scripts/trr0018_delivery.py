"""Package completed TRR-0018 evidence outside the frozen scientific binding."""
from pathlib import Path
import json,hashlib,zipfile,time,subprocess,statistics
from collections import defaultdict
import numpy as np
R=Path(__file__).resolve().parents[1];X=R/'experiments/TRR-0018';OUT=R/'outputs/TRR-0018'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)
def archive(label,paths):
    dest=X/'artifacts'/(label+'.zip');dest.parent.mkdir(exist_ok=True)
    member_receipt=X/'artifacts'/(label+'.json')
    if member_receipt.exists():
        old=json.loads(member_receipt.read_text());assert digest(dest)==old['sha256'];return old
    assert not dest.exists(),'orphan archive; preserve and investigate'
    members=[]
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:
            name=str(p.relative_to(R));members.append({'path':name,'bytes':p.stat().st_size,'sha256':digest(p)});z.write(p,name)
    with zipfile.ZipFile(dest) as z:
        assert len(z.namelist())==len(members)
        for member in members:assert hashlib.sha256(z.read(member['path'])).hexdigest()==member['sha256']
    assert dest.stat().st_size<90_000_000
    entry={'path':str(dest.relative_to(R)),'bytes':dest.stat().st_size,'sha256':digest(dest),'members':members,'every_member_hash_verified':True}
    write(member_receipt,entry);print(label,len(paths),entry['bytes'],flush=True);return entry

def main():
    started=time.time();score=json.loads((X/'score.json').read_text());receipt=json.loads((X/'prediction_receipt.json').read_text())
    matrix=json.loads((X/'canonical_matrix.json').read_text());gate=json.loads((X/'truth_gate.json').read_text())
    assert len(receipt['entries'])==1296 and gate['verified_cells']==1296 and matrix['complete'] and matrix['required_cells']==60
    assert score['prediction_receipt_sha256']==digest(X/'prediction_receipt.json')
    for entry in receipt['entries']:assert digest(R/entry['path'])==entry['sha256']
    archives=[]
    rows=json.loads((OUT/'metadata.json').read_text())
    dev=json.loads((X/'development_freeze.json').read_text())
    for lo in range(0,len(dev['entries']),16):
        paths=[]
        for entry in dev['entries'][lo:lo+16]:
            p=R/entry['path'];assert digest(p)==entry['sha256'];paths += [p,p.with_suffix('.json')]
        archives.append(archive(f'development-{lo//16:02d}',paths))
    by_id=defaultdict(list)
    for entry in receipt['entries']:by_id[entry['id']].append(entry)
    for lo in range(0,len(rows),16):
        paths=[]
        for row in rows[lo:lo+16]:
            for entry in by_id[row['id']]:
                paths += [R/entry['path'],OUT/'receipts'/(row['id']+'__'+entry['method']+'.json')]
        archives.append(archive(f'predictions-{lo//16:02d}',paths))
    inputs=sorted((OUT/'fresh_r4').glob('*'))+sorted((OUT/'resources').glob('*'))+[OUT/'metadata.json',OUT/'evaluator_truth.safetensors']
    archives.append(archive('fresh-inputs-and-evaluator-labels',inputs))
    write(X/'artifact_index.json',{'task_id':'TRR-0018','archives':archives,'count':len(archives),'bytes':sum(a['bytes'] for a in archives),'member_count':sum(len(a['members']) for a in archives),'old352_inputs':'TRR-0016 published observations archive; source hashes in preparation.json','complete_observation_sha256':receipt['binding']['observation_sha256'],'code_weights_credentials_in_archives':False})
    # Timing audit retains every repetition and records paired uncertainty.
    groups=defaultdict(list)
    for entry in receipt['entries']:groups[(entry['setup_id'],entry['condition'],entry['group'],entry['method'])].append(entry)
    timing=[];outliers=[];rng=np.random.default_rng(1818256)
    for setup,condition,group,method in groups:
        if method!='mixed256':continue
        a=groups[(setup,condition,group,method)]
        for comparator in ('frequency256','a1a2'):
            b=groups[(setup,condition,group,comparator)];bm={e['id']:e for e in b};b=[bm[e['id']] for e in a]
            aa=np.array([e['median_seconds'] for e in a]);bb=np.array([e['median_seconds'] for e in b]);indices=rng.integers(0,len(a),(10000,len(a)))
            timing.append({'setup_id':setup,'condition':condition,'group':group,'comparator':comparator,'mean_ratio':float(aa.mean()/bb.mean()),'paired_record_bootstrap95':np.quantile(aa[indices].sum(1)/bb[indices].sum(1),[.025,.975]).tolist(),'median_paired_ratio_posthoc':float(np.median(aa/bb))})
    for key,entries in groups.items():
        med=statistics.median(e['median_seconds'] for e in entries)
        for e in entries:
            if e['median_seconds']>2*med:outliers.append({'id':e['id'],'method':e['method'],'median_seconds':e['median_seconds'],'repetitions':e['phases'],'note':'retained; heuristic timing audit only, no exclusions'})
    write(X/'timing_audit.json',{'paired':timing,'outliers':outliers,'all_raw_repetitions_retained':True,'excluded_runs':[],'bootstrap_seed':1818256,'bootstrap_draws':10000})
    controls=[e for e in receipt['entries'] if e['original_reproduction'] is not None]
    assert len(controls)==704 and all(all(e['original_reproduction'].values()) for e in controls)
    guard=json.loads((X/'prediction_guard.json').read_text());assert guard['returncode']==0 and guard['failure'] is None
    summary=score['summary'];lines=['# TRR-0018: mixed fragment ranking at K256','',
    'The completed voting result is published in [draft PR31](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/31). The research goal remains unfinished. This follow-up combines fragment voting with scores from the existing prefix-derived lookup tables. It still tests exactly 256 candidates through native A2 and trains no extra model.','',
    '## Mechanism and selection','',
    'Preserve the original unique embedding64/intrinsic64 base tokens. Retrieve 128 parents from each existing table, give each distinct parent one vote for each tokenizer suffix, reserve 64 fragment slots for the highest vote counts, then fill remaining slots by the maximum cosine similarity of each fragment in the two tables. Ties retain the original round-robin order. All derived tables come from the same supplied prefix and must be rebuilt after its weights change. Tokenizer metadata is static.','',
    'Ten deterministic candidate-only variants were frozen and scored on 352 opened records. The selected rule improved all four R2/R3 prose conditions, added two Finance proposal hits, and introduced no new omissions or identifier losses. That is retrospective selection evidence. The efficient production code exactly reproduces all 352 selected candidate arrays. The frozen rule was then tested on a new disjoint R4 panel without retuning.','',
    '## Full reconstruction results','',
    '| Panel | Method | Correct / scored | Exact records | Shortlist misses / selector errors | Mean cached seconds/record |','|---|---|---:|---:|---:|---:|']
    for row in summary:
        panel=row['setup_id']+' / '+row['condition']+' / '+row['group']
        lines.append(f"| {panel} | {row['method']} | {row['correct']}/{row['scored']} | {row['exact']}/{row['records']} | {row['scored']-row['proposal_hits']} / {row['wrong_despite_inclusion']} | {row['mean_record_seconds']:.5f} |")
    lines += ['', 'All 432 observations and three current methods were frozen before scoring: 1296 cells, each with three byte-identical repetitions. All 704 unchanged control/input cells reproduce TRR-0016 tokens, ordered candidates, cosine scores and MSE arrays. The 60-cell canonical matrix carries forward 58 source-hashed cells, including the independent TRR-0017 methods, and adds both mixed-method cells. No setup scores are pooled.','',
    '## Paired uncertainty','', '| Panel | Comparator | Correct-token delta | Accuracy delta95% interval | Exact-record delta |', '|---|---|---:|---|---:|']
    for row in score['paired']:
        if row['group']=='stress':continue
        panel=row['setup_id']+' / '+row['condition'];ci=row['token_accuracy_bootstrap95']
        lines.append(f"| {panel} | {row['comparator']} | {row['correct_token_delta']:+d} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {row['exact_record_delta']:+d} |")
    lines += ['', 'Intervals use 10000 paired record bootstrap draws, seed1818256. R2, R3 and both canonical setups are retrospective. R4 is fresh, but a small panel; matching observed scores does not establish statistical equivalence or a universal baseline replacement.','',
    '## Costs, resources and limitations','',
    f"Median prefix-derived table rebuild: {statistics.median(receipt['setup']['prefix_rebuild_seconds']):.4f}s. Tokenizer metadata preparation: {receipt['setup']['tokenizer_suffix_prepare_seconds']:.4f}s. Peak CUDA allocated {receipt['peak_allocated_bytes']/2**30:.3f}GiB, reserved {receipt['peak_reserved_bytes']/2**30:.3f}GiB. Full prediction wall time {(guard['end_unix']-guard['start_unix']):.2f}s. Guard failure: none. Logical A2 cost is 256 simulations per scored token per invocation, plus the common winning-token commit; each invocation is repeated three times for measurement.", '',
    'The reported memory peak is for the comparison process holding prepared assets for all three methods, not a per-method memory comparison. Extra top-k selection, suffix processing and scalar-score transfer add proposal cost. The two full-vocabulary lookup matrix multiplications and native A2 budget remain unchanged. Cached inference includes proposal/input transfer, verification/commit, and output transfer; asset loading, table rebuilding, tokenizer setup and disk I/O are reported separately. A1 historical offline training was not remeasured. Rotating method order and exclusive CUDA access were used. Every raw repetition is retained; timing_audit.json reports possible stalls and paired timing intervals.','',
    'Current canonical executions retain the disclosed batch1 port: only right padding is trimmed, with unchanged native K256 arithmetic and selection. Do not substitute these timings or proposal arrays for historical batch8 native results. The inherited canonical times are not compared with current times.','',
    'The reconstructor uses a supplied public Llama prefix. No experiment here follows genuinely changing recovered-prefix snapshots. The R4 panel uses new books, so it also changes prose distribution; target comparisons remain paired on identical token sequences. Paragraph hashes and first127-token sequences are disjoint from prior project panels, but absence from public-checkpoint pretraining is not established. Source-string recovery for truncated inputs is undefined; decoded-token exactness is separately recorded.','',
    'Fresh sources: [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342) and [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661), public-domain texts with exact download hashes, plus generated identifiers. The evaluator alone uses the pinned rank8/alpha16/seed6012 target LoRA.','',
    'The baseline default remains unchanged and the research goal remains unfinished.','',
    '## Evidence','', '- Manifest: experiments/TRR-0018/manifest.json','- Scores: experiments/TRR-0018/score.json','- Complete matrix: experiments/TRR-0018/canonical_matrix.json','- Raw archives and hashes: experiments/TRR-0018/artifact_index.json','- Reproduction: experiments/TRR-0018/README.md','- Frozen plan: experiments/TRR-0018/BENCHMARK_PLAN.md']
    report=R/'coordination/results/TRR-0018.md';report.write_text('\n'.join(lines)+'\n')
    source=json.loads((X/'committed_source_audit.json').read_text())
    receipts={str(p.relative_to(R)):digest(p) for p in sorted(X.glob('*.json')) if p.name!='manifest.json'}
    manifest={'task_id':'TRR-0018','status':'EXPERIMENT_COMPLETE_RESEARCH_GOAL_OPEN','goal_completed':False,'branch':'task/TRR-0018','request_path':'coordination/requests/TRR-0018.md','result_path':str(report.relative_to(R)),'scientific_execution_commit':source['execution_commit'],'method':'prefix_native_fragment_mixed256','current_methods':['mixed256','frequency256','a1a2'],'observations':432,'prediction_cells':1296,'repetitions':3,'all_repetitions_byte_identical':True,'unchanged_control_cells_reproduced':704,'canonical_comparison_complete':True,'canonical_required_cells':60,'canonical_inherited_cells':58,'binding':receipt['binding'],'setup':receipt['setup'],'summary':summary,'paired':score['paired'],'peak_allocated_bytes':receipt['peak_allocated_bytes'],'peak_reserved_bytes':receipt['peak_reserved_bytes'],'peak_host_rss_bytes':receipt['peak_host_rss_bytes'],'prediction_guard':{k:guard[k] for k in ['command','start_unix','end_unix','returncode','failure']},'truth_status':score['status'],'truth_hashes':score['truth_hashes'],'actual_recovered_prefix_used':False,'fitting_steps':0,'fitting_examples':0,'adaptation_seconds':0,'timing':timing,'timing_outliers':outliers,'receipts':receipts,'artifact_count':len(archives),'artifact_bytes':sum(a['bytes'] for a in archives),'every_archive_member_verified':True,'result_sha256':digest(report),'delivery_script_sha256':digest(Path(__file__)),'previous_result_pull_request':'https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/31','publication_status':'LOCAL_RESULT_READY','packaging_started_unix':started,'packaging_finished_unix':time.time()}
    write(X/'manifest.json',manifest)
    state={'protocol':'TRR-RELAY/1.0','active_task':'TRR-0018','branch':'task/TRR-0018','status':'OPTIMIZATION_VERIFIED_RESEARCH_GOAL_OPEN','request_path':'coordination/requests/TRR-0018.md','result_path':str(report.relative_to(R)),'manifest_path':'experiments/TRR-0018/manifest.json','execution_commit':source['execution_commit'],'canonical_comparison_complete':True,'canonical_required_cells':60,'canonical_matrix_path':'experiments/TRR-0018/canonical_matrix.json','score_status':score['status'],'goal_completed':False,'recovered_prefix_integration':'NOT_RUN_SUPPLIED_PUBLIC_PREFIX','publication_status':'LOCAL_RESULT_READY','previous_result_pull_request':'https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/31'}
    (R/'coordination/STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    print('Packaged and verified',len(archives),'archives; goal remains open',flush=True)
if __name__=='__main__':main()
