"""Independent scalar checks of frozen results and actual executed work, after scoring."""
import argparse,json,statistics
from pathlib import Path
from collections import Counter
from safetensors.torch import load_file
from scripts.agent3_hybrid.common import *
from scripts.agent3_hybrid.evaluate import load_freeze

def main(a):
    f=load_freeze(a.freeze);result=json.loads(Path(a.results).read_text());truth_manifest=json.loads(verify(result['truth']).read_text())
    labels={d:load_file(verify(truth_manifest['domains'][d]['artifact']))['token_ids'].tolist() for d in DOMAINS}
    scored={(c['domain'],c['stage'],c['method']):c for c in result['cells']};checks=[]
    for cell in f['cells']:
        receipt=json.loads(verify(cell['receipt']).read_text());d,s=cell['domain'],cell['stage']
        by_method={m['method']:load_file(verify(m['artifact'])) for m in receipt['methods']}
        for proposer in ('a1','b1'):
            small=by_method[proposer+'_k16'];large=by_method[proposer+'_k256']
            assert (small['candidates']==large['candidates'][:,:,:16]).all()
            assert (small['proposal_scores']==large['proposal_scores'][:,:,:16]).all()
        assert (by_method['b1_alone']['predictions'][:,1:]==by_method['b1_k16']['candidates'][:,:,0]).all()
        for m in receipt['methods']:
            t=load_file(verify(m['artifact']));pred=t['predictions'].tolist();lists=t['candidates'].tolist();c=scored[d,s,m['method']];counts=Counter();first=[]
            for i,row in enumerate(pred):
                previous=False;earliest=None;all_correct=True
                for j in range(1,128):
                    correct=row[j]==labels[d][i][j];included=labels[d][i][j] in lists[i][j-1]
                    counts['scored_tokens']+=1;counts['correct_tokens']+=int(correct);counts['shortlist_omissions']+=int(not included)
                    if not correct:
                        all_correct=False;counts['wrong_despite_inclusion' if included else 'wrong_with_omission']+=1
                        counts['wrong_after_earlier_wrong_commitment']+=int(previous)
                        key=('included_but_wrong_' if included else 'omitted_and_wrong_')+('after_earlier_wrong' if previous else 'before_any_wrong')
                        counts[key]+=1
                        if earliest is None:earliest=j
                    elif previous:counts['correct_after_earlier_wrong']+=1
                    previous=previous or not correct
                counts['exact_clips']+=int(all_correct);first.append(earliest)
            keys=('scored_tokens','correct_tokens','shortlist_omissions','wrong_despite_inclusion','wrong_with_omission','wrong_after_earlier_wrong_commitment','included_but_wrong_after_earlier_wrong','omitted_and_wrong_after_earlier_wrong','included_but_wrong_before_any_wrong','omitted_and_wrong_before_any_wrong','correct_after_earlier_wrong','exact_clips')
            assert all(counts[k]==c[k] for k in keys),(d,s,m['method'],counts)
            assert first==c['first_error_positions']
            medians=[statistics.median(x['measured_seconds']) for x in m['timing_records']]
            assert abs(sum(medians)-c['warmed_total_seconds'])<1e-8
            k=0 if m['method']=='b1_alone' else int(m['method'].split('_k')[1]);runs=[v for record in m['phase_records'] for v in record]
            assert len(runs)==128 and all(len(record)==4 for record in m['phase_records'])
            assert all(v['candidate_simulations']==127*k and v['prefix_commit_tokens']==(128 if k else 0) and v['new_cache_count']==int(k>0) for v in runs)
            checks.append({'domain':d,'stage':s,'method':m['method'],'nested_shortlists_and_b1_rank_one_match':True,'scalar_metrics_match':True,'warmed_medians_match':True,'all_invocation_work_counts_match':True,'invocations':len(runs),'candidate_simulations':sum(v['candidate_simulations'] for v in runs),'committed_cache_tokens':sum(v['prefix_commit_tokens'] for v in runs)})
    write_json(a.output,{'status':'PASS','checked_utc':utc(),'code_commit':commit(),'freeze':binding(a.freeze),'results':binding(a.results),'checks':checks,'scope':'independent scalar metric partitions and work/timing counts; no counterfactual causality claim'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True);main(p.parse_args())
