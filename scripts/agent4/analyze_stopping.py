"""Post-freeze diagnostic only; never selects or reruns a solver policy."""
from common import *
truth=json.loads((OUT/'final/bfloat16/evaluator_truth.json').read_text())
freeze=json.loads((EVID/'final_cpu_adam_r2/freeze.json').read_text())
rows=[]
for f in freeze['prediction_files']:
    assert digest(ROOT/f['path'])==f['sha256']
    record=json.loads((ROOT/f['path']).read_text());y=truth[record['record_id']]
    for trace in record['trace']:
        pos=trace['position'];prior_correct=record['tokens'][:pos]==y[:pos]
        hit=next((j+1 for j,c in enumerate(trace['checks']) if c['token']==y[pos]),None)
        rows.append({'record':record['record_id'],'position':pos,'prior_prefix_correct':prior_correct,'correct':trace['token']==y[pos],'reason':trace['reason'],'genuine_discovery_check':hit,'candidate_checks':len(trace['checks']),'best_mse':trace['best_mse']})
qualified=[r for r in rows if r['prior_prefix_correct'] and r['correct']]
result={'correct_with_correct_prior_prefix':len(qualified),'such_positions_exhausted':sum(r['reason']=='budget_exhausted' for r in qualified),'their_mse_min':min(r['best_mse'] for r in qualified),'their_mse_max':max(r['best_mse'] for r in qualified),'correct_after_wrong_prior_prefix':sum(r['correct'] and not r['prior_prefix_correct'] for r in rows),'rows':rows,'interpretation':'37 genuine-prefix correct decisions exhausted because strict elementwise tolerance does not cover the observed BF16 geometry residual floor. Separately, first-error tokens were never tried despite near-zero genuine candidate residuals. No calibrated policy or new performance claim is produced.'}
write(EVID/'stopping_diagnostic.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
