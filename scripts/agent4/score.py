"""Evaluator-only scoring, after verifying immutable prediction receipts."""
from common import *
import argparse,numpy as np
p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--dtype',default='bfloat16');p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
freezes=[]
for name in args.runs:
    freeze=json.loads((EVID/name/'freeze.json').read_text())
    for f in freeze['prediction_files']:
        assert digest(ROOT/f['path'])==f['sha256']
    freezes.append(freeze)
# First truth opening in this process occurs after every named run is frozen.
truth=json.loads((OUT/args.panel/args.dtype/'evaluator_truth.json').read_text());results={}
for name,freeze in zip(args.runs,freezes):
    rows=[]
    for f in freeze['prediction_files']:
        row=json.loads((ROOT/f['path']).read_text());expected=truth[row['record_id']];pred=row['tokens'];assert len(pred)==len(expected)
        correct=[a==b for a,b in zip(pred,expected)];positions=[]
        for trace in row['trace']:
            pos=trace['position'];found=any(c['token']==expected[pos] for c in trace['checks'])
            positions.append({'position':pos,'correct':correct[pos],'discovered':found,'incorrect_selection_after_discovery':found and not correct[pos],
                'residual':trace['best_mse'],'reason':trace['reason'],'checks':len(trace['checks']),
                'gradients':trace['gradient_steps'],'scans':trace['vocabulary_scans'],'seconds':trace['seconds']})
        first=next((i for i in range(1,len(correct)) if not correct[i]),None)
        rows.append({'record_id':row['record_id'],'group':row['group'],'correct':sum(correct[1:]),'denominator':len(correct)-1,
            'exact':all(correct),'first_error':first,'later_wrong_after_first_error':sum(not c for c in correct[first+1:]) if first is not None else 0,
            'seconds':row['seconds'],'positions':positions,'untraced_abstentions':sum(t==-1 for t in pred)})
    groups={}
    for group in ['ordinary','stress','all']:
        subset=[r for r in rows if group=='all' or r['group']==group];flat=[x for r in subset for x in r['positions']]
        if not subset:continue
        denom=sum(r['denominator'] for r in subset)
        groups[group]={'correct':sum(r['correct'] for r in subset),'denominator':denom,'accuracy':sum(r['correct'] for r in subset)/denom,
            'exact_records':sum(r['exact'] for r in subset),'records':len(subset),'median_seconds':float(np.median([r['seconds'] for r in subset])),
            'p95_seconds':float(np.quantile([r['seconds'] for r in subset],.95)),'total_seconds':sum(r['seconds'] for r in subset),
            'discovered':sum(x['discovered'] for x in flat),'incorrect_selection_after_discovery':sum(x['incorrect_selection_after_discovery'] for x in flat),
            'candidate_checks':sum(x['checks'] for x in flat),'gradient_steps':sum(x['gradients'] for x in flat),'vocabulary_scans':sum(x['scans'] for x in flat),
            'exhausted_tokens':sum(x['reason']=='budget_exhausted' for x in flat),'timed_out_tokens':sum(x['reason']=='token_timeout' for x in flat),
            'abstentions':sum(r['untraced_abstentions'] for r in subset)}
    results[name]={'groups':groups,'records':rows,'freeze_sha256':digest(EVID/name/'freeze.json'),'cost':{k:freeze[k] for k in ['load_seconds','total_process_seconds','peak_allocated','peak_reserved']}}
write(EVID/args.output,{'scored_utc':environment()['utc'],'truth_sha256':digest(OUT/args.panel/args.dtype/'evaluator_truth.json'),'results':results})
print(json.dumps({name:res['groups'] for name,res in results.items()},indent=2))
