"""Render already-scored Stage1 evidence; never changes reconstructions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from scripts.agent3_shortlists.core import BUDGETS, binding, write_json


def percent(x):return f'{100*x:.3f}%'


def main(a):
    r=json.loads(Path(a.results).read_text());d=json.loads(Path(a.drift).read_text());f=json.loads(Path(a.freeze).read_text())
    cells={(c['domain'],c['stage'],c['method']):{m['k']:m for m in c['budgets']} for c in r['cells']}
    lines=['# Frozen B1 shortlists for small-budget A2 under target drift','',
           'The table below answers how often the correct token remained in B1’s top8/16/32 at every tested snapshot. Clip coverage is the fraction of32 clips containing every correct post-BOS token in its fixed lists; it is a perfect-selector upper bound, not achieved A2 reconstruction.','',
           '| Domain | Target stage | Token recall@8 / @16 / @32 | Whole-clip coverage@8 / @16 / @32 |',
           '|---|---:|---|---|']
    for domain in ('pile','finance'):
        for stage in (0,64,128,256):
            c=cells[domain,stage,'b1']
            token=' / '.join(percent(c[k]['token_recall']['estimate']) for k in (8,16,32))
            clip=' / '.join(f"{c[k]['complete_clips']}/32 ({percent(c[k]['clip_coverage']['estimate'])})" for k in (8,16,32))
            lines.append(f'| {domain} | {stage} | {token} | {clip} |')
    lines+=['','Stage0 is the unchanged public base; stage64 is the first fitted compatible LoRA descendant; stages128 and256 continue that same prefix-changing trajectory. This tests one bounded adaptation from the public base, not an independently released full-SFT descendant.','',
            '**Hybrid status:** no small-budget B1+A2 hybrid was tested. The validated available A2 implementation uses untouched public model-prefix weights plus its own reconstructed-token cache. No validated maintained recovered-model-prefix path was available in the inspected/reused assets. Therefore equally accurate faster hybrid recovery and end-to-end target tracking remain untested.','',
            'The smallest empirically promising budget must lose no more than0.1 percentage points of token recall and2 percentage points of whole-clip coverage versus both B1/K256 and A1/K256 in every domain/stage. These limits were committed before outcomes. Passing an empirical limit does not establish equivalence.','',
            '| Domain | Stage | Smallest budget meeting both empirical loss limits |','|---|---:|---:|']
    for c in r['stage1_decision']:lines.append(f"| {c['domain']} | {c['stage']} | {c['smallest'] if c['smallest'] is not None else 'none among8/16/32'} |")
    lines+=['',f"Budgets meeting the empirical limits throughout the matrix: {r['globally_empirically_promising_budgets'] or 'none'}. No confidence fallback, retraining, recalibration, or selective stage omission was used."]
    for domain in ('pile','finance'):
        ordered=[next(x for x in r['stage1_decision'] if x['domain']==domain and x['stage']==s)['smallest'] for s in (0,64,128,256)]
        lines.append(f"For {domain}, the smallest empirically viable list at stages 0 → 64 → 128 → 256 was: "+' → '.join(str(x) if x else 'none ≤32' for x in ordered)+'.')
    comparable=[c for c in r['contrasts'] if c['method']=='b1' and c['k'] in (8,16,32) and c['reference']==f"a1_stage{c['stage']}_k{c['k']}"]
    deltas=[c['token_delta']['estimate']*100 for c in comparable]
    lines.append(f"Across the named cells and small budgets, B1 minus historical A1 token recall ranged from {min(deltas):+.3f} to {max(deltas):+.3f} percentage points. The complete separate comparisons are in the structured result; no cross-snapshot pooled score is used.")
    csv_path=Path(a.results).with_name('shortlist-metrics.csv')
    with csv_path.open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['domain','stage','method','k','records','scored_tokens','included_tokens','omitted_tokens','recall','recall_ci_low','recall_ci_high','complete_clips','clip_coverage','clip_ci_low','clip_ci_high','top1_wrong_tokens','rescued_top1_wrong_tokens','recall_among_top1_wrong'])
        for (domain,stage,method),budgets in sorted(cells.items()):
            for k,c in budgets.items():
                writer.writerow([domain,stage,method,k,c['records'],c['scored_tokens'],c['included_tokens'],c['omitted_tokens'],c['token_recall']['estimate'],*c['token_recall']['paired_source_bootstrap_95'],c['complete_clips'],c['clip_coverage']['estimate'],*c['clip_coverage']['paired_source_bootstrap_95'],c['top1_wrong_tokens'],c['rescued_top1_wrong_tokens'],c['recall_among_top1_wrong']])
    lines+=['','## Complete shortlist matrix','',
            'Each domain has 32 unique paired records and 4,064 scored positions per snapshot. The entries below use K = 1 / 8 / 16 / 32 / 64 / 256 in that order. All omission counts, conditional rescue rates and uncertainty intervals are retained in `shortlist-metrics.csv` beside the structured results.','',
            '| Domain | Stage | Proposer | Token recall (%) at the six budgets | Complete clips out of 32 at the six budgets |',
            '|---|---:|---|---|---|']
    for domain in ('pile','finance'):
        for stage in (0,64,128,256):
            for method in ('b1','a1'):
                c=cells[domain,stage,method]
                recalls=' / '.join(f"{100*c[k]['token_recall']['estimate']:.3f}" for k in BUDGETS)
                clips=' / '.join(str(c[k]['complete_clips']) for k in BUDGETS)
                lines.append(f"| {domain} | {stage} | {method} | {recalls} | {clips} |")
    lines+=['','## Source-level uncertainty and paired effects','',
            'The structured result retains every per-record rank (right-censored beyond256), omission count, paired B1-minus-A1 comparison, and change from stages0 and64. Intervals resample source records jointly with10,000 draws and seed9013. Each domain/stage is reported separately.','',
            'With32 records, even zero records losing candidate inclusion gives a one-sided95% exact upper bound of approximately8.94% on the population probability of any loss in a record. A zero-width bootstrap interval at zero observed losses is not evidence of equivalence. This modest pilot can identify clear omissions; it cannot establish very tight whole-clip noninferiority.','',
            '| Domain | Stage | B1 K32−A1 K256 token recall (pp;95% paired interval) | B1 K32−B1 K256 clip coverage (pp;95% paired interval) |',
            '|---|---:|---|---|']
    for domain in ('pile','finance'):
        for stage in (0,64,128,256):
            def get(ref):return next(c for c in r['contrasts'] if c['domain']==domain and c['stage']==stage and c['method']=='b1' and c['k']==32 and c['reference']==ref)
            x=get(f'a1_stage{stage}_k256')['token_delta'];y=get(f'b1_stage{stage}_k256')['clip_delta']
            def fmt(v):return f"{100*v['estimate']:+.3f} [{100*v['paired_source_bootstrap_95'][0]:+.3f}, {100*v['paired_source_bootstrap_95'][1]:+.3f}]"
            lines.append(f'| {domain} | {stage} | {fmt(x)} | {fmt(y)} |')
    lines+=['','## Actual boundary change','',
            'Relative L2 is computed per record over all127 post-BOS activation vectors and averaged over records. Cosine distance averages post-BOS positions. Values are measured from paired BF16 observations cast toFP64 for this diagnostic.','',
            '| Domain | Stage | Reference stage | Mean relative L2 | Mean cosine distance |','|---|---:|---:|---:|---:|']
    for c in d['comparisons']:
        lines.append(f"| {c['domain']} | {c['stage']} | {c['reference_stage']} | {c['relative_l2_mean']:.6g} | {c['cosine_distance_mean']:.6g} |")
    lines+=['','## Execution costs and reproducibility','',
            'The measured cost below is shortlist production, including deterministic full-vocabulary ranking. It is not A2 reconstruction time. No candidate simulations or prefix-maintenance operations were executed. Scientific rows use the same CPU FP32 record1 path with2threads; TF32 is disabled. Cells may overlap other tasks on the host, so these timings do not establish uncontended cross-system acceleration.','',
            '| Domain | Stage | Proposer | Score seconds | Rank seconds | Transfer/hash seconds | Output I/O seconds |','|---|---:|---|---:|---:|---:|---:|']
    receipts=[]
    for c in f['cells']:
        receipt=json.loads(Path(c['receipt']['path']).read_text());receipts.append(receipt)
        for m in receipt['methods']:
            sums={key:sum(x[key] for x in m['record_timings']) for key in ('scoring_seconds','ranking_seconds','transfer_hash_seconds')}
            lines.append(f"| {c['domain']} | {c['stage']} | {m['method']} | {sums['scoring_seconds']:.3f} | {sums['ranking_seconds']:.3f} | {sums['transfer_hash_seconds']:.3f} | {m['output_io_seconds']:.3f} |")
    lines+=['',f"Total cell wall (including loading): {sum(c['total_seconds'] for c in receipts):.3f}s. Sum of package loading: {sum(c['load_seconds'] for c in receipts):.3f}s. Observation reading/validation: {sum(c['observation_read_validate_seconds'] for c in receipts):.3f}s. Maximum process peak RSS: {max(c['peak_rss_bytes'] for c in receipts)/2**30:.3f}GiB. B1 record forwards:256; A1 record forwards:256; A2 candidate simulations:0.",'',
            'B1 fitting was not repeated. The preserved B1 native fit previously cost2,445.086s externally; that historical fit and its public-data preparation remain required offline resources. Shared target training/capture and backup costs are recorded by TRR-P12 and linked from this task manifest, separately from reconstruction.','',
            'Actual B1 state, readout and package code were copied and hash-verified locally; the independent Windows backup bytes were verified. All package dependencies and all16 candidate outputs are bound. Original candidates/scores/predictions and compressed tracked copies are retained, alongside raw watchdog logs and commands. Truth is evaluator-only and first opened here after the complete matrix freeze.','',
            'B1 state: `088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706`; selected step13000; fixed public readout: `ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1`. Historical A1 uses the pinned public Alpaca affine lens, not an untouched-checkpoint proposer. Its one-record full scores and stable top256 order matched the native path on the preserved fixture.','',
            '## Scope and limitations','',
            'This is a paired shortlist component study on one target trajectory and64 unique source clips shared with Agent2. It is not an independent replication of Agent2. Sources exclude known decoder fitting/selection, prior opened panels, Agent1 reservations, and this trajectory’s target fitting/validation sources through the bound evaluator ledger. Unknown overlap with a public model’s original pretraining corpus is not ruled out.','',
            'The complete dual-canonical reconstruction matrix was not run, and no new active reconstruction replacement is registered. No result here is an overall-best, canonical-replacement, equally accurate hybrid, or end-to-end tracking claim. A negative shortlist budget is a completed Stage1 finding. A positive shortlist budget would only justify a separate maintained-prefix integration study with reserved confirmation sources.','',
            'Reproduction: `experiments/agent3-b1-small-budget-a2/README.md`. Structured evidence: `experiments/agent3-b1-small-budget-a2/manifest.json`. Prospective decisions: `experiments/agent3-b1-small-budget-a2/plan.md`. Failures and repairs are retained in validation receipts; no scientific output was selected from an excluded attempt.']
    Path(a.output).write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('results','drift','freeze','output'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
