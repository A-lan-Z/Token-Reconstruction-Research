"""Separate post-freeze evaluator for the two already-opened root diagnostics."""
from pathlib import Path
import hashlib,json,time
R=Path(__file__).resolve().parents[2];E=R/'experiments/agent4-joint-native-gate/evidence'
OLD=R.parent/'agent4-prefix-only-rescue';O=OLD/'outputs/agent4-prefix-only-rescue/final'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((E/'freeze.json').read_text());outputs=[]
for receipt in freeze['predictions']:
 p=R/receipt['path'];assert digest(p)==receipt['sha256'];outputs.append(json.loads(p.read_text()))
assert len(outputs)==6 and not freeze['truth_read']
assert digest(O/'observations.safetensors')==freeze['observations_sha256']
assert digest(E/'calibration.json')==freeze['calibration_sha256']
# First evaluator-label read in the current component after the complete freeze.
truth=json.loads((O/'evaluator_truth.json').read_text());rows=[]
for result in outputs:
 rid=result['record'];pos=result['position'];gold=truth[rid][pos]
 old=json.loads((OLD/'experiments/agent4-prefix-only-rescue/evidence/final_discrete'/(rid+'.json')).read_text())
 assert result['context']['committed_prefix_ids']==truth[rid][:pos]
 old_ids={c['token'] for c in old['trace'][pos-1]['checks']}
 assert gold not in old_ids
 proposals=set(result['root_proposals']);verified=set(result['root_verified'])
 row={'record':rid,'position':pos,'mode':result['mode'],'gold_token_id':gold,
      'proposed_correct':gold in proposals,'verified_correct':gold in verified,
      'returned_correct':result['returned_block'][0]==gold,'strict_allclose':result['returned_strict_allclose'],
      'unique_root_proposal_count':len(proposals),'new_root_ids_vs_original':sorted(proposals-old_ids),
      'new_correct_roots_vs_original':int(gold in proposals),
      'calibrated_work_seconds':result['budget']['calibrated_work_seconds'],
      'search_seconds':result['budget']['actual_search_seconds'],'work_cap_seconds':result['budget']['work_cap_seconds'],
      'wall_cap_seconds':result['budget']['wall_cap_seconds'],'wall_cap_met':result['budget']['wall_cap_met'],
      'forward_positions':result['forward_positions'],'reverse_position_upper_count':result['reverse_position_upper_count'],
      'vocabulary_rows_scored':result['vocabulary_rows_scored'],'counts':result['budget']['counts'],
      'stop_reason':result['stop_reason'],'total_charged_seconds':result['total_charged_seconds'],
      'root_proposals_not_yet_verified':sorted(proposals-verified)}
 rows.append(row)
summary={}
for mode in ['joint','diagonal','sequential']:
 ms=[r for r in rows if r['mode']==mode]
 total=sum(r['search_seconds'] for r in ms)
 summary[mode]={'correct_proposed':sum(r['proposed_correct'] for r in ms),'correct_verified':sum(r['verified_correct'] for r in ms),
     'correct_returned':sum(r['returned_correct'] for r in ms),'roots':2,'total_search_seconds':total,
     'total_calibrated_work_seconds':sum(r['calibrated_work_seconds'] for r in ms),
     'total_forward_positions':sum(r['forward_positions'] for r in ms),
     'total_reverse_position_upper_count':sum(r['reverse_position_upper_count'] for r in ms),
     'total_vocabulary_rows_scored':sum(r['vocabulary_rows_scored'] for r in ms),
     'correct_proposals_per_search_second':sum(r['proposed_correct'] for r in ms)/total,
     'all_wall_caps_met':all(r['wall_cap_met'] for r in ms)}
paired=[]
for rid,pos in [('natural_11_1',9),('natural_84_1',18)]:
 arms={r['mode']:r for r in rows if r['record']==rid}
 j,d,s=[arms[m] for m in ['joint','diagonal','sequential']]
 paired.append({'record':rid,'position':pos,
    'joint_only_correct_vs_diagonal':int(j['proposed_correct'] and not d['proposed_correct']),
    'diagonal_only_correct_vs_joint':int(d['proposed_correct'] and not j['proposed_correct']),
    'joint_only_correct_vs_sequential':int(j['proposed_correct'] and not s['proposed_correct']),
    'sequential_only_correct_vs_joint':int(s['proposed_correct'] and not j['proposed_correct']),
    'sequential_matches_or_exceeds_recall_at_no_greater_actual_search_cost':s['proposed_correct']>=j['proposed_correct'] and s['search_seconds']<=j['search_seconds'],
    'sequential_no_greater_calibrated_work':s['calibrated_work_seconds']<=j['calibrated_work_seconds']})
stop_no_candidate=summary['joint']['correct_proposed']==0
stop_cost=not summary['joint']['all_wall_caps_met']
stop_control=all(p['sequential_matches_or_exceeds_recall_at_no_greater_actual_search_cost'] for p in paired)
stop=stop_no_candidate or stop_cost or stop_control
reason=('No new useful correct root proposal; wider sequential matches zero recall with lower measured search wall and calibrated work on each root.' if stop_no_candidate and stop_control else 'Stop: joint budget exceeded or wider sequential matches at no greater cost.' if stop else 'Favourable component result only; a separate full-study plan is required.')
result={'decision':'STOP_CURRENT_NATIVE_WINDOW_CONFIGURATION' if stop else 'COMPONENT_FAVOURABLE_ONLY',
        'reason':reason,
        'scope':'Opened native mechanism diagnostics, not confirmation or full reconstruction.',
        'scored_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'freeze_sha256':digest(E/'freeze.json'),
        'truth_sha256':digest(O/'evaluator_truth.json'),'prediction_hashes_verified_before_truth_read':True,
        'rows':rows,'summary':summary,'paired':paired,'new_experiments_authorized_automatically':False}
with (E/'scored_component.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
