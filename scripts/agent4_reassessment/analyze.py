"""Read-only reassessment of frozen rescue evidence; no model execution."""
from pathlib import Path
import json,hashlib,statistics
R=Path(__file__).resolve().parents[2];X=R/'experiments/agent4-holistic-reassessment';E=X/'evidence'
old=R/'experiments/agent4-prefix-only-rescue/evidence'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
score=read(old/'final_score.json')['results']['final_discrete']
truth=read(R/'outputs/agent4-prefix-only-rescue/final/evaluator_truth.json')
rows=[];positions=[];wrong=[];sum_wall=0
for record in score['records']:
    if record['group']!='ordinary':continue
    rid=record['record_id'];saved=read(old/'final_discrete'/(rid+'.json'));gold=truth[rid]
    first=next((i for i in range(1,len(gold)) if saved['tokens'][i]!=gold[i]),None)
    errors=[]
    for t in saved['trace']:
        i=t['position'];right=t['token']==gold[i]
        byloss=sorted(t['checks'],key=lambda c:c['mse'])
        assert t['token']==byloss[0]['token']
        true_checks=[c for c in t['checks'] if c['token']==gold[i]]
        detail={'record':rid,'position':i,'correct':right,'first_error':i==first,'prior_error':first is not None and i>first,'true_proposed':bool(true_checks),'true_mse':true_checks[0]['mse'] if true_checks else None,'chosen_mse':t['best_mse'],'best_second_gap':byloss[1]['mse']-byloss[0]['mse'],'seconds':t['seconds'],'checks':len(t['checks']),'gradients':t['gradient_steps'],'scans':t['vocabulary_scans'],'reason':t['reason']}
        positions.append(detail)
        if not right:wrong.append(detail);errors.append(i)
    rows.append({'record':rid,'correct':record['correct'],'denominator':record['denominator'],'first_error':first,'error_positions':errors,'seconds':record['seconds']})
    sum_wall+=record['seconds']
freeze=read(old/'final_discrete/freeze.json')
for p in freeze['prediction_files']:assert sha(R/p['path'])==p['sha256']
v={'source_commit':'ce04a8dfb11452407df94468964723c6294b8933','source_freeze_sha256':sha(old/'final_discrete/freeze.json'),'records':rows,'errors':wrong,'positions':positions,'first_errors':sum(v['first_error'] for v in wrong),'downstream_errors':sum(v['prior_error'] for v in wrong),'all_best_evaluated_return_invariants_passed':True,'runtime':{'total_seconds':sum_wall,'wrong_position_seconds':sum(v['seconds'] for v in wrong),'wrong_position_fraction':sum(v['seconds'] for v in wrong)/sum_wall,'top7_seconds':sum(sorted([v['seconds'] for v in positions],reverse=True)[:7]),'max_position_seconds':max(v['seconds'] for v in positions),'median_position_seconds':statistics.median(v['seconds'] for v in positions),'min_position_seconds':min(v['seconds'] for v in positions)},'work_counts':{'checks':sum(v['checks'] for v in positions),'gradients':sum(v['gradients'] for v in positions),'scans':sum(v['scans'] for v in positions),'all_positions_exhaust':all(v['reason']=='budget_exhausted' for v in positions)},'prior_recommendations':{'cache':'Executed/qualified numerically; BF16 not identical, no standalone speed win','discrete':'Executed and final-evaluated;85/92 natural,46/46 stress','Gauss_Newton':'Executed after derivative qualification;0/45 development,23.131s; stopped as dominated','candidate_batch8':'Numerical output-equivalence failed; excluded in favor of singleton checks'}}
with (E/'existing_trace_analysis.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:v[k] for k in ['records','errors','runtime','work_counts','prior_recommendations']},indent=2))
