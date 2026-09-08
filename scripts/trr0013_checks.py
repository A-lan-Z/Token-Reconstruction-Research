"""CPU integrity and intervention checks; never reads real evaluation labels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import json,copy,tempfile
import torch
from safetensors.torch import load_file
from scripts.trr0013 import artifact,verify,write_json,save,provenance
from scripts import trr0013_eval as evaluator


def main():
    torch.set_num_threads(1);out=ROOT/'outputs/TRR-0013/cpu_checks_r1';out.mkdir(exist_ok=False)
    prep=json.loads((ROOT/'outputs/TRR-0013/fit_preparation_r1/manifest.json').read_text())
    a=load_file(str(verify(prep['schedules']['ordinary'])));b=load_file(str(verify(prep['schedules']['focused'])));d=load_file(str(verify(prep['difficulty'])))
    assert torch.equal(a['rows'],b['rows']) and torch.equal(a['draws'][:,:256],b['draws'][:,:256])
    assert a['draws'].shape==b['draws'].shape==(6000,512,2)
    for st in [a,b]:
        r=st['rows'].gather(1,st['draws'][:,:,0]);p=st['draws'][:,:,1];assert d['valid'][r,p].all() and (p>0).all()
    for i in [0,1,999,1999,3999,5999]:
        rows=b['rows'][i];valid=d['valid'][rows];eligible=valid.nonzero();values=d['losses'][rows][eligible[:,0],eligible[:,1]]
        ranked=eligible[torch.argsort(values,descending=True,stable=True)[:(len(values)+4)//5]]
        allowed=set(map(tuple,ranked.tolist()));assert all(tuple(x) in allowed for x in b['draws'][i,256:].tolist())
    # Real gate exercised with a fully synthetic four-cell descriptor and a nonexistent truth path.
    contract={'evaluation':{'targets':['public_base','public_lora_2601'],'sources':{'pile':2,'finance':2}}};write_json(out/'contract.json',contract)
    selection={'records':{d:[{'record_id':d+str(i)} for i in range(2)] for d in ['pile','finance']},'curator_payloads':{'pile':{'path':'MUST_NOT_OPEN_TRUTH'},'finance':{'path':'MUST_NOT_OPEN_TRUTH'}}};write_json(out/'selection.json',selection)
    ob=save(out/'obs.safetensors',{'activations':torch.zeros(2,128,2048,dtype=torch.bfloat16),'attention_mask':torch.ones(2,128,dtype=torch.uint8),'position_ids':torch.arange(128).expand(2,-1)})
    cells={d+'__'+t for d in ['pile','finance'] for t in contract['evaluation']['targets']}
    obs={'source_selection':artifact(out/'selection.json'),'observations':{c:ob for c in cells}};write_json(out/'obs.json',obs)
    predictions={}
    for c in cells:
        for m in evaluator.METHODS+['frozen_a1_a2_k256']:
            p=torch.zeros(64 if m=='frozen_a1_a2_k256' else 2,128,dtype=torch.long);p[:,0]=128000
            predictions[c+'::'+m]=save(out/(c+'_'+m+'.safetensors'),{'predictions':p})
    f={'status':'PREDICTIONS_FROZEN_BEFORE_TRUTH','methods':evaluator.METHODS+['frozen_a1_a2_k256'],'contract':artifact(out/'contract.json'),'dependencies':[],'observations':artifact(out/'obs.json'),'cells':{c:{'shape':[2,128],'record_ids':[r['record_id'] for r in selection['records'][c.split('__')[0]]],'observation':ob} for c in cells},'predictions':predictions}
    write_json(out/'valid.json',f);evaluator.validate(out/'valid.json');checks=[]
    for kind in ['missing_method','missing_cell','reordered_sources','modified_code','wrong_dtype','invalid_token','wrong_shape']:
        bad=copy.deepcopy(f);key=sorted(predictions)[0]
        if kind=='missing_method':bad['predictions'].pop(key)
        elif kind=='missing_cell':bad['cells'].pop(sorted(cells)[0])
        elif kind=='reordered_sources':bad['cells'][sorted(cells)[0]]['record_ids'].reverse()
        elif kind=='modified_code':bad['dependencies']=[dict(artifact(__file__),sha256='0'*64)]
        else:
            p=load_file(predictions[key]['path'])['predictions']
            if kind=='wrong_dtype':p=p.int()
            elif kind=='invalid_token':p[0,1]=128256
            else:p=p[:,:127]
            bad['predictions'][key]=save(out/(kind+'.safetensors'),{'predictions':p})
        path=out/(kind+'.json');write_json(path,bad)
        try:evaluator.score(path,out/(kind+'_must_not_exist.json'))
        except (AssertionError,ValueError):checks.append(kind)
        else:raise AssertionError('Gate accepted '+kind)
    write_json(out/'receipt.json',{'status':'PASS','equal_exposures_and_record_batches':True,'first_half_draws_equal':True,'focused_tail_top_quintile':True,'invalid_cases_rejected_before_truth':checks,'sources':[artifact(__file__),artifact(ROOT/'scripts/trr0013_eval.py')],'provenance':provenance()});print('CPU intervention/integrity checks PASS')
if __name__=='__main__':main()
