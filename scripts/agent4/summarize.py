"""Build task-local handoff exclusively from completed, scored receipts."""
from common import *
import statistics
TASK='agent4-prefix-only-inversion'
matched=json.loads((EVID/'final_cpu_score.json').read_text())['results']
static=json.loads((EVID/'static_cpu_score.json').read_text())['results']
p=matched['final_cpu_adam_r2'];a=matched['final_cpu_a1a2'];sp=static['static_cpu_adam'];sa=static['static_cpu_a1a2']
ratio=p['groups']['all']['total_seconds']/a['groups']['all']['total_seconds']

def row(label,x):
    g=x['groups']['all']
    return f"| {label} | {g['correct']}/{g['denominator']} ({100*g['accuracy']:.1f}%) | {g['exact_records']}/{g['records']} | {g['total_seconds']:.2f} | {g['median_seconds']:.2f} / {g['p95_seconds']:.2f} | {g['candidate_checks']} | {g['gradient_steps']} | {g['vocabulary_scans']} |"

def recordrow(label,r):
    first='none' if r['first_error'] is None else str(r['first_error'])
    return f"| {label} / {r['record_id']} | {r['correct']}/{r['denominator']} | {first} | {r['later_wrong_after_first_error']} | {r['seconds']:.2f} |"

def residual_stats(x):
    result={}
    for correct in [True,False]:
        values=[v['residual'] for r in x['records'] for v in r['positions'] if v['correct']==correct]
        result['correct' if correct else 'wrong']={'n':len(values),'min':min(values) if values else None,'median':statistics.median(values) if values else None,'max':max(values) if values else None}
    return result
stopping=json.loads((EVID/'stopping_diagnostic.json').read_text())
first_errors=json.loads((EVID/'first_error_diagnostic.json').read_text())
analysis={'stopping_diagnostic':{k:v for k,v in stopping.items() if k!='rows'},'first_error_diagnostic':first_errors['rows'],'matched_runtime_ratio_prefix_to_a1a2':ratio,'matched_residuals':residual_stats(p),'static_residuals':residual_stats(sp)}
analysis['static_comparability']=json.loads((EVID/'static_comparability.json').read_text())
analysis['matched_quality_target_met']=p['groups']['all']['accuracy']>=.95 and p['groups']['all']['exact_records']/p['groups']['all']['records']>=.5
analysis['matched_cpu_cost_target_met']=ratio<=2
for name,x in {**matched,**static}.items():
    g=x['groups']['all']
    analysis[name]={'vocabulary_rows_scored':g['vocabulary_scans']*128256,'candidate_discovery':g['discovered'],'incorrect_selection_after_discovery':g['incorrect_selection_after_discovery'],'first_errors':[r['first_error'] for r in x['records']],'later_wrong_after_first_error':sum(r['later_wrong_after_first_error'] for r in x['records'])}
write(EVID/'analysis.json',analysis)
verdict='Advance only as a limited CPU component result' if analysis['matched_quality_target_met'] and analysis['matched_cpu_cost_target_met'] else 'Stop the tested bounded variant'
text=f'''# Agent 4 — Prefix-only inversion pilot

1. **Can reconstruction remove the fitted token guesser?** It can recover tokens, but the frozen pilot recovered {p['groups']['all']['correct']}/{p['groups']['all']['denominator']} post-BOS tokens and {p['groups']['all']['exact_records']}/{p['groups']['all']['records']} complete matched-model clips. No A1/B0/B1 or external language model entered the primary solver.
2. **Matching or recovered weights?** Matching public-prefix results are measured below. Static Vikhr mismatch recovered {sp['groups']['all']['correct']}/{sp['groups']['all']['denominator']} tokens. **No legitimately recovered prefix was available or tested.** Static mismatch is not a recovery result.
3. **Is cost competitive?** Matched CPU reconstruction took {ratio:.2f}× the native-policy A1+A2 comparator on the same inputs. These are two-thread CPU measurements; GPU performance was not measured. The quality/cost target was >=95% tokens, >=50% complete clips, and <=2× comparator wall.
4. **Was cold-start tracking demonstrated?** No. This is independent token search using a supplied public prefix, with no model-prefix recovery updates and no tracking sequence. Historical A1 fitting is a comparator preparation cost not remeasured here.
5. **Recommendation:** **{verdict}.** One justified optimizer correction was already tested. These results do not show information loss or rule out other inversion algorithms.

## Method and scope

Started from `5bbc3bf42a81c814404cf84cb46d55f0d3418667` on a separate task branch.
Read and pinned [SipIt v4](https://arxiv.org/abs/2510.15511v4) and its [official implementation](https://github.com/giorgosnikolaou/SIPIT/tree/820683156b7257313046a4fb3c492e52519525b7).
Our budget of 128 trials does not inherit the paper's vocabulary-spanning recovery guarantee.
The initial SGD/snap procedure failed a one-token public CPU diagnostic after 128 checks.
The one corrective variant uses Adam (learning rate 0.01), a float32 provisional embedding cast to BF16 at the input, raw Euclidean vocabulary projection, no snaps/restarts, and at most 128 checks/gradient steps per token. It found that diagnostic token in 17 checks. Model weights stay frozen; only BOS and its own earlier commitments are known. It uses only the current observed activation, commits the best checked MSE candidate on exhaustion, and preserves timed-out suffixes as abstentions.

The source model is public Llama-3.2-1B-Instruct revision `9213176726f574b556790deb65791e0c5aa438b6`; boundary is after blocks 0–3. Forward input embeddings retain their original magnitudes. Full own-prefix recomputation avoids candidate-cache mutation but adds repeated computation. A1+A2 uses its native cached candidate helper and stable top 512 proposal, first 256 candidates, direct cosine and own prior commitments. Its CPU geometry port passed ordered-candidate and prediction equivalence against native `decode_policy` on a public fixture.

## Frozen final measurements

Four public-domain ordinary-text clips and two generated identifier clips, each 16 positions including BOS, were selected after settings freeze. They are unused within this task; this is not a canonical or repository-wide fresh benchmark. All 90 post-BOS tokens remain in the matched denominator. BF16 capture and reconstruction ran on AMD Ryzen 9 9950X3D with two CPU threads. The GPU remained leased to Agent 2; this pilot did not use it.

| Method / observations | Tokens | Complete clips | Total reconstruction s | Median / p95 clip s | Candidate checks | Gradients | Vocabulary scans |
|---|---:|---:|---:|---:|---:|---:|---:|
{row('Prefix-only / matched',p)}
{row('A1+A2 K256 / matched',a)}
{row('Prefix-only / static Vikhr',sp)}
{row('A1+A2 K256 / static Vikhr',sa)}

The two static records are a fixed subset of the six matched records. On that identical subset, the matched primary result is 29/30 tokens and 1/2 clips, versus 30/30 and 2/2 for A1+A2; static accuracy is therefore unchanged for both methods on the common subset. Do not compare 96.7% static with 88.9% full-panel matched as an improvement. Captured boundary drift is measured in `static_comparability.json`.

Tiny empirical p95 values describe this panel only. No records were dropped for exhaustion or timeout.

| Group | Prefix-only matched | A1+A2 matched | Prefix-only static | A1+A2 static |
|---|---:|---:|---:|---:|
'''
for group in ['ordinary','stress']:
    values=[]
    for result in [p,a,sp,sa]:
        g=result['groups'][group];values.append(f"{g['correct']}/{g['denominator']} ({100*g['accuracy']:.1f}%)")
    text+='| '+group+' | '+' | '.join(values)+' |\n'
text+='''
## Failure analysis

| Method / record | Correct tokens | First error position | Later wrong tokens | Reconstruction s |
|---|---:|---:|---:|---:|
'''
for label,result in [('prefix matched',p),('A1+A2 matched',a),('prefix static',sp),('A1+A2 static',sa)]:
    for r in result['records']:text+=recordrow(label,r)+'\n'
text+='\nFirst-error positions are zero-based with BOS at 0. Later errors are reported after wrong commitments; that ordering alone does not establish their individual causes.\n\n'
for label,result in [('matched',p),('static mismatch',sp)]:
    g=result['groups']['all'];rs=residual_stats(result)
    text+=f"On {label}, the correct token was tried at {g['discovered']}/{g['denominator']} positions; {g['incorrect_selection_after_discovery']} final selections were wrong despite discovery. Budget exhaustion occurred at {g['exhausted_tokens']} positions, token timeout at {g['timed_out_tokens']}, and suffix abstention at {g['abstentions']}. Correct/wrong residual summaries are `{json.dumps(rs,sort_keys=True)}`. Residual magnitude is not a correctness certificate.\n\n"
text+='''## Numerical checks, resources and costs

The early synthetic end-to-end test recovered its sequence and verified nonzero gradients, raw continuous/discrete equality and cache isolation. Actual public-prefix CPU diagnostics passed raw-input equality and finite nonzero gradients in FP32 and BF16. Genuine-match residuals, including sequential versus full and cached execution, are retained in `cpu_public_diagnostic.json` and `cpu_cache_qualification.json`; a high-precision diagnostic is not a claim of FP32 reconstruction performance.

The 16-position backward and full-vocabulary scoring qualification measured 2,939,879,424 bytes peak CPU RSS with a 10 GiB cap and >=8 GiB required available host memory. Each final phase has a fail-closed process-group watchdog; primary lifetime kernel VmHWM is separately recorded. Exact commands, start/end times, peak memory and input/output hashes are in the manifest and guard receipts. Vocabulary setup, backward passes, candidate checking and prefix/cache construction are included in reconstruction wall. Per-process wall includes loading and prediction JSON I/O. No prefix-maintenance computation was performed. One-off backup/restore costs and the old comparator-fitting provenance must not be confused with warmed reconstruction.

Actual public prefix, tokenizer, comparator lens and Python dependency archives were backed up, not merely hashed. Independent prefix copies reproduce outputs exactly, and restored dependency imports passed after adding NumPy's sibling shared-library directory. A complete required-distribution archive was additionally restored with Python `-S`, without global site-packages, and reproduced an actual prefix output; see `required_dependency_restore.json`. External interpreter/stdlib, system libraries and OS driver remain runtime requirements. Large assets remain in this task's `outputs` directory; the manifest hashes and locates them.

The stopping rule has a material numerical limitation: 37 of 38 correct decisions with an entirely correct preceding prefix exhausted the budget. Their genuine residuals reached MSE 1.31e-6 because BF16 execution changed with sequence length; elementwise allclose(atol=rtol=1e-5) was too strict. At the five first-error positions, evaluator-only genuine-token checks had MSE 0 to 8.28e-7, while the chosen wrong tokens had MSE 0.0033 to 0.00694. Those genuine tokens had never been tried. Thus both an overly strict verification gate and genuine bounded candidate-discovery failures are present. The observed 80× CPU gap describes this implementation; it is not a lower bound on a corrected solver. Any follow-up should first qualify sequence-length-consistent verification or a numerical tolerance on separate public development material, then use new confirmation inputs. No threshold was tuned on these opened answers.

The common warm-up was one public forward pass. Backward/optimizer initialization was not independently warmed and remains charged to the first primary record. Accordingly, these are post-load reconstruction measurements, not a controlled steady-state throughput benchmark. Retained historical lens fitting cost is unavailable as a directly remeasured preparation phase.

## Stages 2 and 3 limitations

A two-record static fallback uses the existing [Vikhr descendant](https://huggingface.co/Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct/tree/7fa9d06a59246629244cdd3b6b92e4fc756baa0f), cast from its published fp16 weights to BF16 for capture. Tokenizer vocabulary identity and exact target restore were checked by the evaluator. Only the evaluator loads these target weights. Both reconstruction methods still use the untouched public Llama approximation. No target adapter was supplied as a recovery model, and no target was newly trained. The target is a public checkpoint intentionally withheld from the reconstruction code path; it is not cryptographically inaccessible to the shared Unix account.

No validated online model-prefix recovery implementation/state was located in the pinned resources. Agent 2 was asked for qualified assets and provenance; no such recovery state was integrated. Consequently, the required public-versus-recovered comparison and own-history closed-loop Stage3 were not run. No cold-start tracking claim is made.

## Evidence, deviations and handoff

Read `experiments/agent4-prefix-only-inversion/manifest.json` for exact evidence paths and hashes; replay commands are in `experiments/agent4-prefix-only-inversion/REPRODUCE.md`. The request was saved verbatim at `coordination/requests/agent4-prefix-only-inversion.md`. Access/cost interpretation is in `ACCESS_AND_COST.md`.

Predictions and traces are create-only and frozen before scoring. Capture, prediction and scoring use separate code paths/processes on one account; missing bwrap prevented OS-level isolation. This is explicitly not a sealed canonical benchmark. The single-token public diagnostics capture and score in one process after writing predictions and are development only.

Retained failures: initial missing-bwrap read attempts; dependency restore missing numpy.libs, then corrected; the stronger isolated restore exposed missing idna in OS package dependency metadata, repaired with an actual-module supplement; test import path failure, then 21 reused tests passed; initial SGD search failure; and a prematurely launched final prediction process terminated before scoring because its explicit 16-position qualification was not yet recorded. The aborted attempt is excluded as an orchestration error; the identical six-record panel was rerun after qualification with no setting changes or dropped records. Its approximately one-minute elapsed cost was not separately instrumented. Four new bounded-search/timeout invariant tests passed. The initial tiny smoke/asset-copy checks occurred before their first implementation commit; later scientific runs record exact full commits in their receipts.

The canonical dual-benchmark matrix is **NOT RUN / COMPARISON INCOMPLETE**. Method registration is task-local under the packet's ban on global-registry changes. No P03 holdout data was accessed, no PR was merged, and no other workspace was modified.
'''
report=ROOT/'coordination/results'/f'{TASK}.md';report.parent.mkdir(exist_ok=True)
report.write_text(text)
# Manifest indexes compact tracked evidence and the actual local preserved assets.
artifacts=[]
for f in sorted((ROOT/'experiments'/TASK).rglob('*')):
    if f.is_file() and 'upstream/SIPIT/' not in str(f) and f.name!='manifest.json':
        artifacts.append({'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':digest(f)})
large=[]
for directory in ['backup','restore','evaluator_only/static_backup','evaluator_only/static_restore','final/bfloat16','static/bfloat16']:
    for f in sorted((OUT/directory).glob('*')):
        if f.is_file():large.append({'path':str(f),'bytes':f.stat().st_size,'sha256':digest(f)})
manifest={'task_id':TASK,'status':'COMPLETE_BOUNDED_MATCHED_AND_STATIC_PILOT_NO_RECOVERY','base_commit':'5bbc3bf42a81c814404cf84cb46d55f0d3418667','source_commit':environment()['commit'],'created_utc':environment()['utc'],'result_path':str(report.relative_to(ROOT)),'request_path':f'coordination/requests/{TASK}.md','request_sha256':digest(ROOT/'coordination/requests'/f'{TASK}.md'),'frozen_settings':json.loads((EVID/'final_settings_freeze.json').read_text()),'analysis':analysis,'matched_results':matched,'static_results':static,'artifacts':artifacts,'actual_local_assets':large,'phase_commits':{name:json.loads((EVID/name/'freeze.json').read_text())['environment']['commit'] for name in ['final_cpu_adam_r2','final_cpu_a1a2','static_cpu_adam','static_cpu_a1a2']},'canonical_comparison_complete':False,'recovered_prefix_tested':False,'cold_start_tracking_demonstrated':False,'gpu_used':False,'no_pr_merges':True,'no_global_registry_changes':True,'no_p03_data_access':True}
write(ROOT/'experiments'/TASK/'manifest.json',manifest)
state_path=ROOT/'coordination/STATE.json';state=json.loads(state_path.read_text())
state.update({'status':manifest['status'],'updated_utc':manifest['created_utc'],'frozen_method_id':'agent4_prefix_only_adam128','execution_commit':json.loads((EVID/'final_cpu_adam_r2/freeze.json').read_text())['environment']['commit'],'score_status':'TASK_LOCAL_UNUSED_SOURCES_FROZEN_BEFORE_SCORING','pull_request':{'task_pr_status':'PENDING_PUBLICATION'},'pull_request_url':None})
state['agent4'].update({'status':manifest['status'],'gpu_lease':'not used; CPU-only pilot','analysis':analysis,'stages2_3':'static public-surrogate component complete; recovered prefix and closed-loop tracking NOT RUN','canonical_matrix':'NOT_RUN_COMPARISON_INCOMPLETE'})
state={k:state[k] for k in ['active_task','branch','status','updated_utc','frozen_method_id','execution_commit','score_status','pull_request','pull_request_url','last_accepted_task','request_path','result_path','manifest_path','canonical_comparison_complete','canonical_matrix_status','agent4']}
state.update({'protocol':'TRR-RELAY/1.0','base_commit':manifest['base_commit'],'prior_state_available_at_commit':manifest['base_commit'],'phase_commits':manifest['phase_commits'],'execution_commit_scope':'Final matched prefix-only prediction phase; other phase commits are listed explicitly.','score_receipt_path':'experiments/agent4-prefix-only-inversion/evidence/final_cpu_score.json','static_score_receipt_path':'experiments/agent4-prefix-only-inversion/evidence/static_cpu_score.json','charter_sha256':digest(ROOT/'RESEARCH_CHARTER.md')})
state_path.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({'report':str(report),'artifacts':len(artifacts),'local_assets':len(large),'ratio':ratio,'verdict':verdict}))
