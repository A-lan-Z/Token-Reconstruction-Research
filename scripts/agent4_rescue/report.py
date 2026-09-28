"""Generate the final measured handoff after all output sets freeze."""
from shared import *
def read(p):return json.loads(Path(p).read_text())
final=read(E/'final_score.json')['results'];dev=read(E/'development_score.json')['results']
qual=read(E/'executor_qualification.json');fail=read(E/'post_freeze_failure_diagnostic.json')
methods=['original','cached','discrete','a1a2']
def g(m,group='ordinary'):return final['final_'+m]['groups'][group]
d=g('discrete');a=g('a1a2');o=g('original')
advance=d['correct']/d['denominator']>=.95 and d['exact']/d['records']>=.5 and d['seconds']<=2*a['seconds']
def table(group):
    lines=['| Method | Tokens | Exact clips | Reconstruction s | Median / p95 s | Candidate forwards | Gradients | Vocab scans |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        v=g(m,group)
        lines.append(f"| {m} | {v['correct']}/{v['denominator']} | {v['exact']}/{v['records']} | {v['seconds']:.3f} | {v['median_seconds']:.3f}/{v['p95_seconds']:.3f} | {v['checks']} | {v['gradients']} | {v['scans']} |")
    return '\n'.join(lines)
costlines=['| Method | Load/setup s | Post-import process s | Peak allocated/reserved GiB | Peak host GiB |','|---|---:|---:|---:|---:|']
for m in methods:
    c=final['final_'+m]['cost'];costlines.append(f"| {m} | {c['setup_seconds']:.3f} | {c['process_seconds']:.3f} | {c['peak_allocated']/2**30:.3f}/{c['peak_reserved']/2**30:.3f} | {c['peak_host_rss']/2**30:.3f} |")
frontier={}
for group in ['ordinary','stress']:
    frontier[group]=[m for m in methods if not any(g(n,group)['correct']>=g(m,group)['correct'] and g(n,group)['exact']>=g(m,group)['exact'] and g(n,group)['seconds']<=g(m,group)['seconds'] and (g(n,group)['correct']>g(m,group)['correct'] or g(n,group)['exact']>g(m,group)['exact'] or g(n,group)['seconds']<g(m,group)['seconds']) for n in methods if n!=m)]
summary={'task_id':'agent4-prefix-only-rescue','final':{m:final['final_'+m]['groups'] for m in methods},'development':{m:dev['development_'+m]['groups'] for m in methods+['inverse']},'natural_discrete_to_a1a2_wall_ratio':d['seconds']/a['seconds'],'natural_original_to_discrete_speed_ratio':o['seconds']/d['seconds'],'advance_gate_met':advance,'frontier':frontier,'inverse_disposition':'Stopped after dominated development result; not run on final','canonical_matrix':'NOT RUN / COMPARISON INCOMPLETE','recovered_prefix_tested':False,'cold_start_tracking':False}
write(E/'summary.json',summary)
fp=[v for v in qual['rows'] if v['dtype']=='torch.float32'];bf=[v for v in qual['rows'] if v['dtype']=='torch.bfloat16']
report=f"""# Agent 4: bounded prefix-only rescue

Discrete search recovered **{d['correct']}/{d['denominator']} natural-text tokens**, **{d['exact']}/{d['records']} exact clips**, in **{d['seconds']:.3f}s**. A1+A2 recovered **{a['correct']}/{a['denominator']}**, **{a['exact']}/{a['records']} exact clips**, in **{a['seconds']:.3f}s**. Discrete search is {o['seconds']/d['seconds']:.2f} times faster than the original schedule on these natural clips, but costs {d['seconds']/a['seconds']:.2f} times A1+A2.

The predeclared advance gate is **{'MET' if advance else 'NOT MET'}**: >=95% natural-token recovery, >=50% exact clips and <=2 times A1+A2 wall. **{'Further scoped mismatch evaluation is required.' if advance else 'Stop this bounded configuration; do not advance to mismatched/recovered-prefix experiments.'}** This does not reject prefix-only inversion generally. No fitted proposer enters either challenger.

## Historical anchor and bottleneck

The historical corrected pilot remains29/30 in9.824s versus30/30 in1.688s. Original bugged CPU results remain separate and unchanged in PR29. Rescue starts from eac972712b655cbb3712b72641614ff41c907483, preserving corrected result commit76c30972aec91aa4aa9c22b2b68135e033fc3642 and scientific execution commit ae5275561a6dd74c7e07ecbe93f26648c887c693.

The new original-schedule replay reproduced every candidate order and final token:29/30 in10.899s versus30/30 in1.782s. Its only error, stress position15, was never proposed; earlier tokens were correct and no timeout/final-return bug occurred. The genuine full-prefix residual at this error is MSE {fail['historical_failed_token_genuine_full_prefix_mse']:.3g}. That position contributes17.2% of its clip, not a dominant single-position total.

A separately synchronized diagnostic measured5.117s continuous forward/backward,2.276s discrete verification,1.363s vocabulary ranking,0.040s table setup and0.030s committed-prefix tensor setup. The remaining1.839s combines optimizer/scalar reads/guards/loop overhead. These clocks are intrusive; only uninstrumented runs support speed ratios. The remainder is not an isolated synchronization measurement.

The local code already avoids per-token table clones, per-step full vocabulary difference tensors, unused suffix/head execution, token decoding and hot-loop garbage collection. Full-prefix recomputation and repeated scalar reads were actually present.

## Qualified executor and numerical limitation

Earlier layer K/V are detached constants. Each trial creates temporary cache state; only the chosen token commits once. Parameter-version changes fail closed until caches are invalidated and rebuilt. Raw embedding magnitudes, cut4, native FP32 rotary construction and BF16 observations are preserved.

Real Llama checks cover positions1,4,15,39,127. FP32 maximum relative forward/gradient errors are {max(v['forward_relative_error'] for v in fp):.3g}/{max(v['gradient_relative_error'] for v in fp):.3g}; BF16 maxima are {max(v['forward_relative_error'] for v in bf):.3g}/{max(v['gradient_relative_error'] for v in bf):.3g}. The cached arm is a **numerical execution variant**, not a byte-identical optimization. Trial caches remained unchanged. Batch8 checks differed from singleton execution, so batching was excluded for challengers. Native comparator batching remains separately qualified.

Cached Adam returns the same29/30 historical tokens but needs3840 checks/gradients and38.758s. All30 positions exhaust the unchanged strict verifier. This is a negative execution-control result; candidate trajectories and stopping changed. No threshold was loosened to make known answers pass.

## Methods and development

- Original: corrected Adam128, lr0.01, seed4401, raw Euclidean projection and full-prefix recomputation.
- Cached: identical Adam schedule/tolerance, current-token numerical executor.
- Discrete: one random public token start;8 rounds of8 proposals, singleton actual-forward verification and best-so-far retention. Half mean-squared activation-error gradients define a quadratic substitution surrogate. Initial trust radius is twice RMS raw embedding norm; damping adapts to actual versus predicted improvement. No fitted initialization or fallback.
- Inverse:4 local steps, each4 CG iterations plus curvature estimate, with matrix-free JVP/VJP, damping and actual-forward step acceptance. Math attention is used for derivatives; numerical differences are recorded. No dense Jacobian.
- A1+A2: supplied historical public Alpaca lens, stable native top512/first256 candidates, direct cosine and own committed tokens. Comparator only.

New search arms use cached row norms, vocabulary matrix-vector scoring, direct distances for32 shortlisted rows and token-ID tie breaking before selecting8. This still reads the entire vocabulary. Mean-loss normalization is included in curvature/radius scaling; Adam's rate was not blindly multiplied.

Two generated public prose clips and one identifier clip formed development: original/cached41/45, discrete40/45 (natural30/30, stress10/15), inverse0/45, A1+A2 45/45. All five outputs froze before scoring. Inverse took23.131s versus discrete12.088s and was stopped before final selection. Its adjoint check passed, but that did not ensure useful token proposals. The post-freeze failure diagnostic reproduces its first-position candidate order and records actual continuous-step improvements. No parameter sweep or second rescue followed.

## Final natural panel

Four unused task-local public-domain paragraphs,24 positions including BOS:

{table('ordinary')}

## Separate unusual-token panel

Two generated identifier clips,24 positions including BOS:

{table('stress')}

Every record/token remains in its denominator, including exhausted guesses or abstentions. Natural and stress are not pooled for the advance decision. The empirical frontier under token count, exact clips and runtime is {frontier}; it describes this small panel only.

## Costs and verification

{chr(10).join(costlines)}

Reconstruction includes table preparation, all trial forwards, derivatives, cache commits, trust control, synchronization and Python work. Cache arms additionally execute one commit per emitted token including BOS. Full-prefix arms redo earlier positions on every trial. Inverse traces separately count JVP/VJP products; loss-gradient VJPs are included in VJP totals and must not be counted twice.

Setup/process clocks start after imports; the outer watchdog includes interpreter/import time. Process clocks include prediction serialization. One public forward warmup is shared; optimizer/backward startup is charged to reconstruction. This is not a steady-state benchmark. Historical lens fitting is supplied and not remeasured: this is a supplied-prefix component comparison, not cold-start cost.

Hardware: RTX5080,16GiB, two CPU threads. Largest-workspace qualification included127 prior positions, derivatives and full FP32 vocabulary scoring. Limits:6GiB reserved GPU, >=2GiB free GPU, >=8GiB host available, <=10GiB child host RSS, <80C, <=1800s jobs. Every matrix guard passed. Actual independent prefix/tokenizer/lens copies and fresh required-dependency extraction were verified before final capture; Python-S with no global site-packages reproduced a prefix output exactly. Interpreter/stdlib, system libraries and driver remain external.

Four new cache/return invariant tests and six supplied analytical checks passed. Supplied synthetic checks demonstrate algebra only; real Llama qualification and performance receipts are separate.

## Scope and release

Only BOS, reconstructed earlier tokens, current activation and fixed public assets inform search. No model update occurs within a record. Capture, prediction and score are separate processes on one account, not cryptographic isolation. Final methods froze before source selection; all four outputs froze before truth opening. Source byte hashes, paragraph indices and seed947201 are retained. No repository-wide unseen-text or pretraining-disjointness claim is made.

Public model: Llama-3.2-1B-Instruct revision9213176726f574b556790deb65791e0c5aa438b6, blocks0-3. No actual target prefix substitutes for recovered weights. No qualified Agent3 hybrid was integrated. GPU coordination/release is documented; no other task files changed.

Canonical dual-benchmark matrix: **NOT RUN / COMPARISON INCOMPLETE**. No P03, paid compute, target training, PR merge or global registry change. No recovered-prefix or cold-start result.

Result: coordination/results/agent4-prefix-only-rescue.md.
Manifest: experiments/agent4-prefix-only-rescue/manifest.json.
Reproduction: experiments/agent4-prefix-only-rescue/REPRODUCE.md.
Brief: coordination/requests/agent4-prefix-only-rescue.md; supplied background is preserved under incoming. Its inaccessible-commit statement is historical; PR29 is now public. Recommendations were tested, not treated as measured local findings.

New public release is not inferred from the pilot approval. A sanitized package/list is prepared locally. Supplied documents, coordination data, absolute runtime paths, actual model/dependency assets and evaluator truth are excluded. No rescue branch was pushed or rescue PR created.
"""
path=ROOT/'coordination/results/agent4-prefix-only-rescue.md';path.write_text(report)
(X/'REPRODUCE.md').write_text("""# Reproduction

Use exact commits in phase freeze receipts; Python3.12.3, torch2.10.0+cu128, transformers5.3.0.
From root set OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src.
All output names are create-only; rerun in a fresh output root, never overwrite.

Read PLAN.md and freeze.json. Run in order:
1. run.py --method original --name replay_original; then profile/replay_profile and a1a2/replay_a1a2.
2. qualify.py; inverse_preflight.py.
3. prepare.py --panel development; matrix.py --panel development.
4. restore.py creates actual backup/restore copies and freshly extracts dependency archives.
5. Freeze the retained settings; prepare.py --panel final originally selected source bytes after freeze.
6. matrix.py --panel final --methods original cached discrete a1a2.
7. Only after every final output is frozen: failure_diagnostic.py.
8. PYTHONPATH=src python3 -m pytest -q tests/test_agent4_rescue.py.

These scripts are under scripts/agent4_rescue. Wrap GPU matrices with the inherited
scripts/agent4/watchdog.py --receipt NEW_PATH --timeout 1800 -- python3 SCRIPT...
Coordinate a GPU lease; exact guard commands and full commits are in receipts.
For exact replay use retained final_sources.json rather than selecting new sources.
Keep BF16 observations and the numerical executor classification unchanged.

The shared loader initially resolves the inherited pilot outputs through a read-only
symlink. New independent backups and restored dependencies live in
outputs/agent4-prefix-only-rescue; backup_restore.json indexes actual assets/hashes.
When relocating, recreate the documented input layout or point the loader at these
backups. No private target weights or fitted initialization is required; only the
A1+A2 comparator loads the supplied historical public lens.
""")
manifest={'task_id':'agent4-prefix-only-rescue','status':'COMPLETE_LOCAL_RELEASE_NOT_AUTHORIZED','source_commit':environment()['commit'],'request_path':'coordination/requests/agent4-prefix-only-rescue.md','request_sha256':digest(ROOT/'coordination/requests/agent4-prefix-only-rescue.md'),'result_path':str(path.relative_to(ROOT)),'result_sha256':digest(path),'summary':summary,'phase_commits':{f.parent.name:read(f)['environment']['commit'] for f in E.glob('*/freeze.json')},'resource_guards':{p.name:read(p) for p in E.glob('*guard.json')},'backup_restore':read(E/'backup_restore.json'),'canonical_comparison_complete':False,'new_publication_authorized':False,'no_p03':True,'no_paid_compute':True,'no_pr_merge':True,'artifacts':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(X.rglob('*')) if p.is_file() and p.name!='manifest.json']}
write(X/'manifest.json',manifest)
state=read(ROOT/'coordination/STATE.json');state.update(status='RESCUE_COMPLETE_LOCAL_RELEASE_NOT_AUTHORIZED',updated_utc=environment()['utc']);state['rescue_summary']=summary;state['publication']={'status':'SANITIZED_RELEASE_PREPARED_APPROVAL_REQUIRED'}
(ROOT/'coordination/STATE.json').write_text(json.dumps(state,indent=2)+'\n')
print('Report assembled. Advance gate:',advance)
