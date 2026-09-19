"""Package completed TRR-0019 evidence; never reads evaluator labels."""
from pathlib import Path
import json,hashlib,zipfile,time,subprocess
ROOT=Path(__file__).resolve().parents[1];X=ROOT/'experiments/TRR-0019';OUT=ROOT/'outputs/TRR-0019'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2)
def main():
    score=json.loads((X/'score.json').read_text());freeze=json.loads((X/'prediction_freeze.json').read_text())
    matrix=json.loads((X/'canonical_matrix.json').read_text());assert matrix['complete'] and matrix['required_cells']==62
    for name,sha in freeze['binding']['sources'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('frozen source changed: '+name)
    reg=json.loads((X/'registry.json').read_text())
    for name,key in [('registry.json','registry_sha256'),('canonical_matrix.json','matrix_sha256'),('score.json','score_sha256')]:
        if digest(X/'inherited'/name)!=reg['inherited_sources'][key]:raise RuntimeError('inherited evidence changed')
    assert len(freeze['entries'])==1360
    for entry in freeze['entries']:
        if digest(ROOT/entry['path'])!=entry['sha256']:raise RuntimeError('prediction changed')
    artifacts=X/'artifacts';artifacts.mkdir(exist_ok=False);archive=None;archives=[];index=[];seen={};raw_bytes=0
    paths=sorted((OUT/'predictions').glob('*.safetensors'))+sorted((OUT/'receipts').glob('*.json'))
    for p in paths:
        sha=digest(p);raw_bytes+=p.stat().st_size
        if sha not in seen:
            if archive is None or archive.fp.tell()>64*2**20:
                if archive is not None:archive.close()
                dest=artifacts/f'raw-{len(archives):02d}.zip';archives.append(dest)
                archive=zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED,compresslevel=6)
            member=str(p.relative_to(ROOT));archive.write(p,member)
            seen[sha]={'archive':str(dest.relative_to(ROOT)),'member':member}
        index.append({'path':str(p.relative_to(ROOT)),'sha256':sha,'bytes':p.stat().st_size,**seen[sha]})
    if archive is not None:archive.close()
    for p in archives:
        with zipfile.ZipFile(p) as z:
            for member in z.namelist():
                actual=hashlib.sha256(z.read(member)).hexdigest()
                if seen.get(actual)!={'archive':str(p.relative_to(ROOT)),'member':member}:raise RuntimeError('archive verification failed')
    write(X/'artifact_index.json',{'entries':index,'logical_files':len(index),'unique_payloads':len(seen),'raw_bytes':raw_bytes,
       'archives':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in archives],
       'deduplication':'Identical raw bytes stored once. Restore each logical path from its recorded archive/member and verify SHA256.'})
    summaries=score['summary'];pairs=score['paired'];lines=['# TRR-0019: faster A2 at the same candidate budget','',
      "Exact GPU replay reduces the existing no-A1 method's full inference time by 37.84% on Pile and 35.94% on Finance, preserving every output. Applied to A1+A2 itself, replay reduces time by 40.37% and 38.56%. This is a useful A2 execution improvement at the same candidate budget. A separate shared-context attention kernel reaches 43.68% and 45.43% reductions for the no-A1 method, at the cost of three additional Finance token errors (among 13,990 tokens) and two fewer exact Finance rows. Pile and both R4 prose/identifier conditions retain the original no-A1 scores.",'',
      'The exact replay controls preserve every token, ordered candidate, cosine score and MSE value for both A1+A2 and the existing no-A1 mixed method on all 272 inputs. Shared attention has different floating-point scores and is evaluated as a separate method. The table reports the mean of three-repeat median inference times per input.','',
      '| Setup / condition / group | Implementation | Correct / scored | Exact inputs | Seconds / input |',
      '|---|---|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['setup_id']} / {s['condition']} / {s['group']} | {s['method']} | {s['correct']}/{s['scored']} | {s['exact']}/{s['records']} | {s['seconds_per_record']:.5f} |")
    lines+=['','`a1_native`: current A1+A2 control. `a1_graph`: exact GPU replay. `mixed_native`: frozen TRR-0018 no-A1 proposer with native A2. `mixed_graph`: exact replay of that method. `mixed_fast`: shared-context attention plus replay.','',
      '## Paired execution savings','',
      '| Setup / condition / group | Comparison | Runtime reduction | Runtime ratio 95% interval | Correct-token change |',
      '|---|---|---:|---|---:|']
    for s in pairs:
        if (s['method'],s['control']) not in [('a1_graph','a1_native'),('mixed_graph','mixed_native'),('mixed_fast','mixed_native')]:continue
        interval=s['runtime_ratio_interval95'];lines.append(f"| {s['setup_id']} / {s['condition']} / {s['group']} | {s['method']} / {s['control']} | {100*(1-s['runtime_ratio']):.2f}% | [{interval[0]:.4f}, {interval[1]:.4f}] | {s['correct_delta']:+d} |")
    setup=freeze['setup'];cold=json.loads((X/'attention_qualification.json').read_text())['compile_and_first_run_seconds']
    lines+=['','Intervals use 10,000 paired record bootstrap draws, seed 1919256. Three repetitions per cell and rotating method order reduce timing noise; these intervals do not cover every possible device/workload condition. Full paired accuracy intervals and record changes are in score.json. Scores are never pooled across setups.','',
      'Both methods retain all 256 candidates and the separate native batch1 winning-token commit. No new predictor is fitted. Exact replay is the preferred implementation when preserving the original result is required; shared attention is an optional numerical trade-off. Candidate-free reconstruction remains unsolved.','',
      '## Preparation, memory and prefix changes','',
      f"Prefix loading: {setup['prefix_load']:.3f}s; tokenizer/suffix metadata: {setup['tokenizer_suffix']:.3f}s; derived lookup tables: {setup['prefix_table_rebuild']:.3f}s; A1 asset preparation: {setup['a1_asset_load']:.3f}s. Historical A1 fitting is not remeasured.",
      '',f"Exact replay warmup/capture: {setup['exact_replay']['warmup_seconds']:.3f}/{setup['exact_replay']['capture_seconds']:.3f}s; fast replay warmup/capture: {setup['fast_replay']['warmup_seconds']:.3f}/{setup['fast_replay']['capture_seconds']:.3f}s. These main-run figures reuse compiled kernels. The initial fast-attention probe spent {cold:.3f}s in compilation plus first inference; that cold cost must not be hidden.",
      '',f"Whole-process peak CUDA allocated/reserved: {freeze['peak_allocated']/2**30:.3f}/{freeze['peak_reserved']/2**30:.3f} GiB; host RSS: {freeze['peak_host_rss']/2**30:.3f} GiB. All control assets and both replay engines were resident, so these are not standalone per-method peaks.",
      '','Inference includes proposal/input transfer, verification/commit, and output transfer. Disk IO and preparation are recorded separately. Replay is most useful when setup is amortized across inputs; compare total preparation plus inference for short-lived jobs. Tokenizer metadata and kernel compilation can be reused across weight updates. Lookup tables must be rebuilt when their prefix changes.',
      '','The capture reads existing weight storage on every replay. An in-place public-weight change and restoration passed the exact-output check without recapture. Replacing tensor storage or changing geometry requires a new capture. This is an execution check, not an actual recovered-prefix training trajectory; reconstruction benchmarks still use the supplied public prefix.','',
      '## Scientific scope and evidence','',
      'All 1,360 cells (five implementations by 272 inputs) froze before the current scorer opened source labels. All panels were already opened in earlier tasks; these results are retrospective. The canonical matrix contains 62 cells: 60 inherited with source hashes and both cells for the new numerical variant. Inherited timings are not compared with current timings.',
      '','Canonical controls retain the disclosed batch1/right-padding-trim port. Native controls reproduce their archived anchors exactly. Exact replay optimizes execution of those frozen rules; shared attention is separately registered because floating-point arithmetic changes.',
      '','Fresh exclusive-compute checks and a largest 128-token qualification preceded the matrix. The prediction process checkpoints cells and fails closed on source changes, orphan outputs, resource limits, competing GPU processes, and nonrepeatable results. The scorer rejects missing cells, changed outputs, and changed bindings before labels are opened.',
      '','Independent dense-math checks passed at past lengths 1, 39 and 127. Shared history stayed unchanged and changing one candidate did not affect the others. All 1,088 saved native/replay control outputs were also compared with archived predictions as raw tensor bytes and matched exactly. The kernel comparison is within the declared BF16 tolerance, not byte equivalence to native SDPA.',
      '','No shortlist policy or candidate budget was tuned here. Shared attention was selected from truthless execution qualifications and stayed frozen through scoring. Its numerical differences are preserved. The native baseline default remains unchanged.',
      '',f"Execution commit: `{freeze['environment']['commit']}`. Raw evidence contains {len(index):,} logical files, archived with exact-byte deduplication and verified hashes. Reproduction: experiments/TRR-0019/README.md. Mechanism: experiments/TRR-0019/MECHANISM.md. Manifest: experiments/TRR-0019/manifest.json. Full metrics: experiments/TRR-0019/score.json. Complete matrix: experiments/TRR-0019/canonical_matrix.json.",
      '', '## Publication status','',
      'This follow-up is committed locally. Public publication remains blocked by the earlier automatic approval review, which rejected uploading the research/evidence payload without explicit authorization for the public destination. No alternate upload or push retry was attempted.']
    report=ROOT/'coordination/results/TRR-0019.md';report.write_text('\n'.join(lines)+'\n')
    (X/'DRAFT_PR.md').write_text('A2 repeats launch and context-copying work for every token. This change records its native GPU execution and adds a separately evaluated shared-context attention implementation, retaining all 256 candidates and the original separate winner commit. No new predictor is trained.\n\nValidation: full 1,360-cell retrospective comparison, three repetitions per cell, both canonical setups, frozen outputs, exact replay equality, weight-refresh checks, guarded resources, and the complete 62-cell active-method matrix. See coordination/results/TRR-0019.md and experiments/TRR-0019/manifest.json. Candidate-free reconstruction and actual recovered-prefix trajectory integration remain unsolved.\n')
    evidence=[p for p in X.glob('*.json') if p.name!='manifest.json']+list(X.glob('*.md'))+[report,Path(__file__),ROOT/'tests/trr0019_kernel_math.py']
    manifest={'task_id':'TRR-0019','execution_commit':freeze['environment']['commit'],'handoff_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'environment':freeze['environment'],
      'inherited_TRR0018_full_commit':subprocess.check_output(['git','rev-parse','7a65a0a'],cwd=ROOT,text=True).strip(),'model':'meta-llama/Llama-3.2-1B-Instruct','model_revision':'9213176726f574b556790deb65791e0c5aa438b6',
      'binding':freeze['binding'],'start_utc':freeze['start_utc'],'end_utc':freeze['end_utc'],'setup':setup,
      'metrics_path':'experiments/TRR-0019/score.json','canonical_complete':True,'canonical_cells':62,
      'commands':['PYTHONPATH=src python3 scripts/trr0017/exclusive_watchdog.py --receipt experiments/TRR-0019/benchmark_guard.json --timeout 7200 -- python3 scripts/trr0019/benchmark_predict.py','PYTHONPATH=src python3 scripts/trr0019/benchmark_score.py'],
      'evidence_sha256':{str(p.relative_to(ROOT)):digest(p) for p in evidence},
      'failures_or_exclusions':[{'attempt':'direct shared attention as an exact replacement','disposition':'Not byte-identical; excluded from exact claims and separately registered/evaluated.'},
          {'attempt':'First packaging-script construction','disposition':'Outer Python quoting syntax error before a file was written; inference experiment unaffected. Corrected using a literal PowerShell here-string.'}],
      'truth_status':'Retrospective; complete current freeze before scorer access','actual_recovered_prefix_used':False,
      'publication_status':'PRIOR_AUTOMATIC_APPROVAL_REVIEW_BLOCK_PERSISTS'}
    write(X/'manifest.json',manifest)
    state=json.loads((ROOT/'coordination/STATE.json').read_text());state.update(status='A2_ACCELERATION_VERIFIED_LOCAL_PUBLICATION_PENDING',
      canonical_comparison_complete=True,canonical_matrix_path='experiments/TRR-0019/canonical_matrix.json',canonical_required_cells=62,
      score_status=score['status'],updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      achieved_scope='A2 efficiency; candidate-free reconstruction remains unsolved',recovered_prefix_integration='IN_PLACE_PUBLIC_WEIGHT_REFRESH_ONLY')
    (ROOT/'coordination/STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'logical_files':len(index),'unique_payloads':len(seen),'archive_bytes':sum(p.stat().st_size for p in archives),'report':str(report)},indent=2))
if __name__=='__main__':main()