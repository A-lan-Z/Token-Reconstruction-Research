"""Package the frozen TRR-0016 result; no candidate or decision changes."""
from pathlib import Path
import sys,json,hashlib,time,zipfile,statistics
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/trr0016'))
from support import *
sys.path.insert(0,str(ROOT/'scripts/trr0016'))
from score import evaluator_truth


def pack(paths,path):
    members=[]
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in paths:
            rel=str(p.relative_to(ROOT));archive.write(p,rel)
            members.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest(p)})
    with zipfile.ZipFile(path) as archive:
        for member in members:
            assert hashlib.sha256(archive.read(member['path'])).hexdigest()==member['sha256']
    assert path.stat().st_size<100_000_000,'split evidence below GitHub single-file limit'
    return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path),'members':members}


def main():
    score=json.loads((X/'score.json').read_text());matrix=json.loads((X/'canonical_matrix.json').read_text())
    receipt=json.loads((X/'prediction_receipt.json').read_text());guard=json.loads((X/'prediction_guard.json').read_text())
    assert guard['returncode']==0 and guard['failure'] is None and len(receipt['entries'])==1408 and matrix['complete']
    check_binding(receipt['binding'])
    rows=json.loads((OUT/'metadata.json').read_text());truth,truth_hashes=evaluator_truth(rows)
    assert truth_hashes==score['truth_hashes']
    truth_path=OUT/'evaluator_truth.safetensors';save_file(truth,str(truth_path),metadata={'truth_opened_after':'experiments/TRR-0016/truth_gate.json'})
    destination=X/'evidence';destination.mkdir(exist_ok=False)
    fresh=OUT/'fresh_r3'
    archives=[pack([fresh/'observations.safetensors',fresh/'metadata.json',fresh/'evaluator_truth.json',fresh/'sources.json',OUT/'metadata.json',truth_path],destination/'fresh-panel-and-evaluator-labels.zip')]
    diagnostic_paths=list((OUT/'development').glob('*.safetensors'))+list((OUT/'orderings').glob('*.safetensors'))
    archives.append(pack(diagnostic_paths,destination/'exploratory-candidate-arrays.zip'))
    for start in range(0,len(rows),16):
        paths=[];part=rows[start:start+16]
        for row in part:
            for method in METHODS:
                stem=row['id']+'__'+method
                paths.extend([OUT/'predictions'/(stem+'.safetensors'),OUT/'receipts'/(stem+'.json')])
        archives.append(pack(paths,destination/f'predictions-{start:03d}-{start+len(part)-1:03d}.zip'))
    prior=ROOT/'experiments/TRR-0015/evidence/inputs-and-evaluator-labels.zip'
    index={'task_id':'TRR-0016','archives':archives,'total_archive_bytes':sum(a['bytes'] for a in archives),'member_count':sum(len(a['members']) for a in archives),'every_member_hash_verified':True,
        'prior_input_archive':{'path':str(prior.relative_to(ROOT)),'sha256':digest(prior)},'combined_inputs_reproduction':'Restore archived paths into the named worktrees; scripts/trr0016/prepare.py combines prior272 and fresh80 observations. Current combined observation hash is in prediction_receipt.json.'}
    write(X/'artifact_index.json',index)
    setups=[('original_r2','matched','natural','Development prose / matched'),('original_r2','lora256','natural','Development prose / LoRA'),
        (CANONICAL[0],'clean_pile_lora_cut4','canonical','Canonical Pile'),(CANONICAL[1],'historical_finance_cut4','canonical','Canonical Finance'),
        ('fresh_r3','matched','natural','Fresh prose / matched'),('fresh_r3','lora256','natural','Fresh prose / LoRA'),
        ('original_r2','matched','stress','Development identifiers / matched'),('original_r2','lora256','stress','Development identifiers / LoRA'),
        ('fresh_r3','matched','stress','Fresh identifiers / matched'),('fresh_r3','lora256','stress','Fresh identifiers / LoRA')]
    lookup={(r['setup_id'],r['condition'],r['group'],r['method']):r for r in score['summary']}
    quality=['| Setup | Frequency K256 | Original K256 | A1+A2 K256 | Original K512 |','|---|---:|---:|---:|---:|']
    costs=['| Setup | Frequency K256 | Original K256 | A1+A2 K256 | Original K512 | Frequency / baseline |','|---|---:|---:|---:|---:|---:|']
    methods=('frequency256','fragment256','a1a2','fragment512')
    for setup,condition,group,label in setups:
        rr=[lookup[(setup,condition,group,m)] for m in methods]
        quality.append('| '+label+' | '+' | '.join(f"{100*r['accuracy']:.4f}%; {r['exact']}/{r['records']} exact" for r in rr)+' |')
        costs.append('| '+label+' | '+' | '.join(f"{r['mean_record_seconds']:.4f}" for r in rr)+f" | {rr[0]['seconds']/rr[2]['seconds']:.3f}x |")
    setup=receipt['setup'];build=statistics.median(setup['prefix_rebuild_seconds'])
    cache_bytes=sum(setup['cache'][k] for k in ('transform_bytes','embedding_cache_bytes','intrinsic_cache_bytes'))
    anchors=[e for e in receipt['entries'] if e['original_reproduction'] is not None]
    assert len(anchors)==816 and all(e['original_reproduction']['tokens'] and e['original_reproduction']['candidates'] for e in anchors)
    report=f'''# TRR-0016: suffix-frequency ordering at K256

The previously completed result is published in draft PR #30: https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/30. This follow-up investigates further optimization. The research goal remains unfinished.

The refinement improves the original K256 method on both canonical setups and on fresh prose. On fresh matched prose, token errors fall from29 to10 and exact clips rise from15/32 to24/32. On fresh shifted prose, errors fall from25 to8 and exact clips rise from18/32 to25/32. A1+A2 still makes only2 and3 errors respectively (30/32 and29/32 exact), so the improvement does not establish a baseline replacement. Fresh paired bootstrap intervals for the improvement over original K256 exclude zero in both prose conditions; the matched-prose deficit to A1+A2 also excludes zero. Canonical gains over A1+A2 remain uncertain.

## Fixed mechanism
Keep every deduplicated embedding64/intrinsic64 base proposal from the same supplied prefix. Count how many distinct base proposals produce each tokenizer suffix. Fill the remaining slots with the most frequently suggested suffixes; break ties by the original round-robin order. A2 verifies exactly256 candidates using its own reconstructed history. No new model, learned parameters, training examples, fitting steps or target-prefix access are introduced. The prefix-derived catalogs and their rebuild rules are unchanged.

An exact-context priority diagnostic rescued no omitted tokens. Six fixed tokenizer-only ordering alternatives were then compared on opened R2 candidate recall. Frequency ordering rescued7/13 matched and11/13 shifted omissions with no new omissions. The efficient implementation reproduced all80 exploratory candidate arrays exactly. This development selection is separate from the fresh confirmation below.

## Quality
All1408 current cells (352 observations x4 methods) were frozen before the scorer loaded truth. Three byte-identical repetitions per cell. The fresh R3 panel consists of32 disjoint prose clips and8 new generated identifier clips, paired across matched and the same trained LoRA target. R2 and both canonical setups remain retrospective. No method was changed after the fresh panel was opened.

{chr(10).join(quality)}

Percentages are post-BOS token accuracy. Exact requires every scored token to match. All denominators, decoded-token exactness, proposal recall and errors despite candidate inclusion appear in score.json. Source-string recovery is not inferred from token-truncated inputs. Paired record bootstrap intervals use10000 draws with seed1616256; nonsignificance is not evidence of equivalence. Do not pool scores across setups.

## Runtime and preparation
{chr(10).join(costs)}

Seconds per input: median of three synchronized runs, averaged over each setup. All compared methods were rerun on the same RTX5080 with the same timing boundary and rotating method order. Historical A1 offline training cost was not remeasured. The frequency calculation is included in proposal time. Prefix-dependent catalog rebuild median {build:.4f}s; tokenizer-only preparation {setup['tokenizer_suffix_prepare_seconds']:.4f}s; derived GPU catalogs {cache_bytes/1e9:.3f}GB. Rebuild whenever prefix weights change. Method-specific phase and I/O timings are retained in receipts.

One timing outlier is retained in the primary means: original K256 on fresh shifted book84_13 took5.913,6.747 and0.841 seconds in its three identical-output repetitions. The cause is unknown; no resource guard was breached. Consequently, its apparent mean-time disadvantage to the new method in that panel must not be interpreted as a method speedup. A supplemental post-hoc median of paired per-input ratios gives new/original K256=1.012 there. All raw timings remain unchanged; see timing_audit.json. Across fresh prose and canonical cells, the new method's mean cached time is about1.1-3.3% above A1+A2. Prefix-rebuild cost is separate.

## Comparability and validation
The full matrix contains54 canonical cells:52 inherited frozen results with hashes and2 new frequency-method cells. All four current controls were run in both canonical setups, both development target conditions, and both fresh target conditions. Every one of816 unchanged control/input cells reproduced the prior returned tokens and ordered candidate arrays.

The canonical current runs retain the same single-record ports as TRR-0015, removing only right padding and adapting serialization. They are not exact historical batch8 executions. The inherited numerical caveat remains: TRR-0015 baseline candidate sets differed from the historical native reference at3 Pile and208 Finance positions, although every returned token matched. Current cross-method timing is compared only within the shared geometry.

Seven focused tests passed. Largest128-position/K512 resource qualification and native-compatible output checks passed. The scorer verifies matrix completeness, code/input/asset/output hashes, geometry, finite values, IDs and argmax decisions, and rejects missing cells, changed predictions and changed code. The independent provenance audit checks fresh observations, truth, metadata and selection hashes against their original capture before scoring.

Scientific code commit: {setup['environment']['commit']}.
Prediction command: env PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/agent4/watchdog.py --receipt experiments/TRR-0016/prediction_guard.json --timeout 5400 -- python3 scripts/trr0016/predict.py
Runtime: {guard['end_unix']-guard['start_unix']:.3f}s; peak allocated {receipt['peak_allocated_bytes']/2**30:.3f}GiB; reserved {receipt['peak_reserved_bytes']/2**30:.3f}GiB. Minimum sampled GPU free {min(s['gpu_free_mib'] for s in guard['samples'])}MiB; maximum temperature {max(s['temperature_c'] for s in guard['samples'])}C. No resource failure in the full matrix.

Preparation failures are preserved in attempts.json: an evaluator import error and insufficient unused Alice paragraphs in the original middle-half window. Both occurred before any fresh capture artifacts. The final source window widened symmetrically to the middle80% of both books while preserving exclusions,16 clips per book and the frozen method. The failed context diagnostic and all six exploratory orders are preserved.

## Limits and artifacts
These tests still use a supplied public four-layer prefix, not an evolving recovered prefix. Fresh confirmation covers new records for the same two target conditions, not a newly trained target update. The baseline default is unchanged and the research goal is not marked complete.

Result: coordination/results/TRR-0016.md
Manifest: experiments/TRR-0016/manifest.json
Full scores and uncertainty: experiments/TRR-0016/score.json
Canonical matrix: experiments/TRR-0016/canonical_matrix.json
Evidence index: experiments/TRR-0016/artifact_index.json
New archives: {len(archives)}; {index['total_archive_bytes']:,} bytes; {index['member_count']} verified members. Prior272 observations are referenced by their already-published archive hash; fresh observations and all current predictions are newly archived.
'''
    result_path=ROOT/'coordination/results/TRR-0016.md';result_path.write_text(report)
    manifest={'task_id':'TRR-0016','branch':'task/TRR-0016','status':'EXPERIMENT_COMPLETE_RESEARCH_GOAL_OPEN','goal_completed':False,
        'request_path':'coordination/requests/TRR-0016.md','result_path':str(result_path.relative_to(ROOT)),
        'scientific_execution_commit':setup['environment']['commit'],'methods':list(METHODS),'candidate_budgets':BUDGETS,
        'observations':352,'prediction_cells':1408,'repetitions':3,'canonical_comparison_complete':True,'canonical_required_cells':54,'canonical_inherited_cells':52,'canonical_new_cells':2,
        'summary':score['summary'],'paired':score['paired'],'preparation':setup,'binding':receipt['binding'],'truth_hashes':truth_hashes,
        'scoring_seed':1616256,'bootstrap_draws':10000,'fitting_steps':0,'fitting_examples':0,'adaptation_seconds':0,'actual_recovered_prefix_used':False,
        'truth_status':'canonical and R2 retrospective; freshR3 complete matrix frozen before first score; no retuning after reveal',
        'hardware_and_dependencies':setup['environment'],'peak_allocated_bytes':receipt['peak_allocated_bytes'],'peak_reserved_bytes':receipt['peak_reserved_bytes'],'peak_host_rss_bytes':receipt['peak_host_rss_bytes'],
        'prediction_start_unix':guard['start_unix'],'prediction_end_unix':guard['end_unix'],'prediction_wall_seconds':guard['end_unix']-guard['start_unix'],
        'guard_command':guard['command'],'guard_failure':guard['failure'],'preparation_attempts':json.loads((X/'attempts.json').read_text()),
        'timing_outlier':'original fragment256, fresh shifted book84_13; all repetitions preserved; see timing_audit.json; no speedup claim over original K256',
        'unchanged_control_cells_reproduced':816,'canonical_ports':'same record-batch1 ports as TRR-0015; native candidate-array caveat preserved',
        'source_string_recovery':'undefined for truncated panels; not equated with decoded-token accuracy','artifact_index_sha256':digest(X/'artifact_index.json'),
        'evidence_archive_count':len(archives),'evidence_archive_bytes':index['total_archive_bytes'],'all_archive_members_verified':True,
        'previous_result_pull_request':'https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/30','delivery_script_sha256':digest(Path(__file__))}
    manifest['receipts']={str(p.relative_to(ROOT)):digest(p) for p in sorted(X.glob('*.json'))}
    write(X/'manifest.json',manifest)
    state=json.loads((ROOT/'coordination/STATE.json').read_text());state.update(status='EXPERIMENT_COMPLETE_RESEARCH_GOAL_OPEN',updated_utc=utc(),execution_commit=manifest['scientific_execution_commit'],canonical_comparison_complete=True,score_status=manifest['truth_status'],goal_completed=False)
    (ROOT/'coordination/STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'archives':len(archives),'bytes':index['total_archive_bytes'],'members':index['member_count'],'result':str(result_path),'manifest':str(X/'manifest.json')},indent=2),flush=True)
if __name__=='__main__':main()