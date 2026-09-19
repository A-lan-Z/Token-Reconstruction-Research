from support import *
from collections import defaultdict
import copy

def main():
    f=json.loads((X/'freeze.json').read_text());outputs=gate(f);negative=[]
    for mode in ['missing_cell','changed_output','changed_source']:
        bad=copy.deepcopy(f)
        if mode=='missing_cell':bad['entries'].pop()
        elif mode=='changed_output':bad['entries'][0]['sha256']='0'*64
        else:bad['binding']['prefix_sha256']='0'*64
        try:gate(bad)
        except ValueError:negative.append(mode)
    if len(negative)!=3:raise RuntimeError('negative gate failed')
    n.write(X/'truth_gate.json',{'negative_rejections':negative,'cells':768,'freeze_sha256':n.digest(X/'freeze.json'),'truth_opened_after_unix':time.time()})
    truthpath=INPUT/'evaluator_truth.safetensors';truth=n.load_file(str(truthpath));rows=[];groups=defaultdict(list)
    for e in f['entries']:
        if e['method']=='a1_native':continue
        key=e['id'],e['method'];a=outputs[key];b=outputs[e['id'],'a1_native'];target=truth[e['id']]
        ac=a['tokens'][1:]==target[1:];bc=b['tokens'][1:]==target[1:]
        row={k:e[k] for k in ['id','setup_id','method']};row.update(novel_correct=int(ac.sum()),baseline_correct=int(bc.sum()),scored=len(ac),novel_mean_error=float(a['cosine_error'].mean()),baseline_mean_error=float(b['cosine_error'].mean()))
        difference=(a['tokens'][1:]!=b['tokens'][1:]).nonzero().flatten()
        row['first_disagreement']=None
        if len(difference):
            index=int(difference[0]);delta=float(a['cosine_error'][index]-b['cosine_error'][index])
            relation='lower' if delta< -1e-6 else 'higher' if delta>1e-6 else 'within_tolerance'
            row['first_disagreement']={'position':index+1,'baseline_correct':bool(bc[index]),'novel_correct':bool(ac[index]),'common_prefix_all_correct':bool(bc[:index].all()),'novel_error':float(a['cosine_error'][index]),'baseline_error':float(b['cosine_error'][index]),'delta':delta,'novel_loss_relation':relation}
        rows.append(row);groups[e['setup_id'],e['method']].append(row)
    summary=[]
    for (setup,method),values in sorted(groups.items()):
        first=[v['first_disagreement'] for v in values if v['first_disagreement'] is not None]
        harmed=[v for v in first if v['baseline_correct'] and not v['novel_correct'] and v['common_prefix_all_correct']]
        worse=[v for v in values if v['novel_correct']<v['baseline_correct']]
        summary.append({'setup_id':setup,'method':method,'records':len(values),'novel_correct':sum(v['novel_correct'] for v in values),'baseline_correct':sum(v['baseline_correct'] for v in values),'accuracy_worse_records':len(worse),'of_worse_novel_mean_error_lower':sum(v['novel_mean_error']<v['baseline_mean_error']-1e-6 for v in worse),'first_disagreements':len(first),'first_wrong_with_correct_common_prefix':len(harmed),'first_wrong_loss_relation':{k:sum(v['novel_loss_relation']==k for v in harmed) for k in ['lower','within_tolerance','higher']}})
    n.write(X/'score.json',{'scope':'retrospective diagnostic, native batch1 whole-sequence forward; output tokens unchanged; not a new reconstruction or timing comparison','first_loss_tolerance':1e-6,'freeze_sha256':n.digest(X/'freeze.json'),'truth_sha256':n.digest(truthpath),'per_record':rows,'summary':summary})
    for v in summary:print(json.dumps(v),flush=True)
if __name__=='__main__':main()
