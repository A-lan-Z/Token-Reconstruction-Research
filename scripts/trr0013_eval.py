"""Frozen standalone package, observation-only predictions, and gated paired scoring."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import argparse,json,time,shutil,inspect,subprocess,os
import numpy as np
import torch
from safetensors.torch import load_file
from scripts.trr0013 import artifact,verify,write_json,save,Guard,provenance,load_p09_fixed_state,_predict_row_package,utc
from scripts.trr0004_fresh_confirmation import run_warmed_prediction
from scripts.trr0010_analysis import paired_exact_cp
METHODS=['frozen_B1','ordinary_continuation','focused_continuation']


def code_bindings():
    # Bind the full small local scientific source tree rather than guessing imported dependencies.
    files=sorted((ROOT/'src/token_reconstruction').rglob('*.py'))+sorted((ROOT/'scripts').rglob('*.py'))
    return [artifact(p) for p in files]


def package(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);inputs=json.loads((ROOT/'experiments/TRR-0013/inputs.json').read_text());old=json.loads(verify(inputs['package_manifest']).read_text())
    descriptors={'frozen_B1':{'state':inputs['starting_state'],'loader_kwargs':old['methods']['expanded_fixed']['loader_kwargs']}}
    for arm,name in [('ordinary','ordinary_continuation'),('focused','focused_continuation')]:
        r=json.loads((ROOT/f'outputs/TRR-0013/fit_{arm}_r1/receipt.json').read_text());descriptors[name]=r['selected']
    result={}
    for name,d in descriptors.items():
        p=out/(name+'.safetensors');shutil.copyfile(verify(d['state']),p);assert artifact(p)['sha256']==d['state']['sha256']
        result[name]={'state':artifact(p),'loader_kwargs':d['loader_kwargs']}
    p=out/'readout.safetensors';shutil.copyfile(verify(inputs['readout']),p)
    code=out/'code';code.mkdir();shutil.copytree(ROOT/'src/token_reconstruction',code/'token_reconstruction',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'scripts/trr0010_p09_fixed_loader.py',code/'trr0010_p09_fixed_loader.py')
    fn=inspect.getsource(_predict_row_package)
    (code/'predict.py').write_text('import torch\nSTORED_SEQUENCE_TOKENS=128\nSCORED_POST_BOS_TOKENS=127\nVOCABULARY_SIZE=128256\nBOS_TOKEN_ID=128000\nPackageError=ValueError\n'+fn)
    shutil.copyfile(ROOT/'scripts/trr0013_restore_smoke.py',out/'restore_smoke.py')
    write_json(out/'package.json',{'restore_driver':artifact(out/'restore_smoke.py'),'task_id':'TRR-0013','methods':result,'readout':artifact(p),'code':[artifact(p) for p in sorted(code.rglob('*.py'))],'contract':artifact(ROOT/'experiments/TRR-0013/contract.json'),'created_utc':utc(),'source':artifact(__file__),'provenance':provenance()})
    print(json.dumps(artifact(out/'package.json')),flush=True)


def load_package(path):
    m=json.loads(Path(path).read_text());verify(m['contract'])
    for b in m['code']:verify(b)
    assert list(m['methods'])==METHODS
    return m


def predict(package_path,observation_path,output,a1=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);guard=Guard(seconds=2400);clock=time.perf_counter();obs=json.loads(Path(observation_path).read_text());verify(obs['source_selection']);contract=json.loads(verify(obs['contract']).read_text())
    expected={d+'__'+t for d in ['pile','finance'] for t in contract['evaluation']['targets']};assert set(obs['observations'])==expected
    m=load_package(package_path);results={};cold={}
    names=['frozen_a1_a2_k256'] if a1 else METHODS
    for name in names:
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        if a1:
            from scripts import trr0004_predict_confirmation as legacy,trr0003_footing_compare as footing
            desc=json.loads((ROOT/'experiments/TRR-0010/setup/a1_a2_public_runtime_descriptor_v1.json').read_text())
            for key in ['public_reference','retained_a1_lens','public_embedding_table']:
                b=dict(desc[key]);b['path']=str(ROOT/b['path']);verify(b)
            for b in desc['model_snapshot']['files'].values():verify(dict(b,path=b['snapshot_path']))
            for key,b in desc['native_code_bindings'].items():
                if key!='trr0010_runner':verify(dict(b,path=str(ROOT/b['path'])))
            precut,lens,E,load_evidence=legacy._load_public_prefix(snapshot=Path(desc['model_snapshot']['path']),reference_path=Path(desc['public_reference']['path']),lens_path=ROOT/desc['retained_a1_lens']['path'],embedding_path=Path(desc['public_embedding_table']['path']),device=torch.device('cuda'))
            adapter=legacy._A2Adapter(precut=precut,lens=lens,embeddings=E,device=torch.device('cuda'),policy=footing._fixed_k256_policy())
        else:
            d=m['methods'][name];model=load_p09_fixed_state(verify(d['state']),**d['loader_kwargs']).to('cuda').eval();E=load_file(str(verify(m['readout'])))['embeddings'].to('cuda')
            adapter=lambda h,mask,pos:_predict_row_package(model,E,h,mask,pos,device=torch.device('cuda'))
        cold[name]={'load_seconds':time.perf_counter()-start};guard.check()
        for cell,b in obs['observations'].items():
            t=load_file(str(verify(b)));domain=cell.split('__')[0];n=64 if a1 else contract['evaluation']['sources'][domain]
            assert set(t)=={'activations','attention_mask','position_ids'} and t['activations'].shape==(contract['evaluation']['sources'][domain],128,2048)
            assert t['activations'].dtype==torch.bfloat16 and torch.isfinite(t['activations']).all() and t['attention_mask'].all() and torch.equal(t['position_ids'],torch.arange(128).expand(len(t['position_ids']),-1))
            if a1:adapter.begin_cell()
            def guarded(h,mask,pos):
                guard.check();return adapter(h,mask,pos)
            pred,timing=run_warmed_prediction(observations=t['activations'][:n],attention_mask=t['attention_mask'][:n],position_ids=t['position_ids'][:n],predictor=guarded,device='cuda',warmup_runs=1,measured_runs=3)
            assert all(x['repeated_prediction_exact'] for x in timing['records'])
            pb=save(out/(cell+'__'+name+'.safetensors'),{'predictions':pred})
            row={'prediction':pb,'timing':timing,'observation':b,'resources':guard.check()}
            if a1:row['candidate_simulations_all_four_calls']=adapter.candidate_simulations
            write_json(out/(cell+'__'+name+'.receipt.json'),{'cell':cell,'method':name,**row,'package':artifact(package_path),'observations':artifact(observation_path),'source':artifact(__file__),'provenance':provenance()})
            results[cell+'::'+name]=row;print(json.dumps({'cell':cell,'method':name,'records':n,'seconds':timing['total_elapsed_seconds']}),flush=True)
        if a1:del adapter,precut,lens,E
        else:del adapter,model,E
        torch.cuda.empty_cache()
    write_json(out/'receipt.json',{'task_id':'TRR-0013','predictions':results,'cold':cold,'package':artifact(package_path),'observations':artifact(observation_path),'dependencies':code_bindings(),'elapsed_seconds':time.perf_counter()-clock,'resources':guard.check(),'truth_opened':False,'provenance':provenance()})


def freeze(package_path,observation_path,standalone,anchor,out):
    m=load_package(package_path);obs=json.loads(Path(observation_path).read_text());contract=json.loads(verify(obs['contract']).read_text());selection=json.loads(verify(obs['source_selection']).read_text())
    receipts=[json.loads(Path(p).read_text()) for p in [standalone,anchor]];pred={};deps=code_bindings()+[artifact(package_path),artifact(observation_path),obs['source_selection'],obs['contract']]+[artifact(p) for p in [standalone,anchor]]+[d['state'] for d in m['methods'].values()]+[m['readout']]+m['code']
    for r in receipts:
        assert r['package']==artifact(package_path) and r['observations']==artifact(observation_path)
        for b in r['dependencies']:verify(b)
        pred.update({k:v['prediction'] for k,v in r['predictions'].items()})
    expected={c+'::'+name for c in obs['observations'] for name in METHODS+['frozen_a1_a2_k256']};assert set(pred)==expected
    cells={c:{'shape':[contract['evaluation']['sources'][c.split('__')[0]],128],'record_ids':[r['record_id'] for r in selection['records'][c.split('__')[0]]],'observation':b} for c,b in obs['observations'].items()}
    descriptor=ROOT/'experiments/TRR-0010/setup/a1_a2_public_runtime_descriptor_v1.json';d=json.loads(descriptor.read_text());deps.append(artifact(descriptor))
    for key in ['public_reference','retained_a1_lens','public_embedding_table']:
        b=dict(d[key]);b['path']=str(ROOT/b['path']);deps.append(b)
    for b in d['model_snapshot']['files'].values():deps.append(dict(b,path=b['snapshot_path']))
    write_json(out,{'task_id':'TRR-0013','status':'PREDICTIONS_FROZEN_BEFORE_TRUTH','methods':METHODS+['frozen_a1_a2_k256'],'cells':cells,'predictions':pred,'dependencies':deps,'observations':artifact(observation_path),'contract':obs['contract'],'created_utc':utc(),'provenance':provenance()})
    validate(out)


def validate(path):
    f=json.loads(Path(path).read_text());assert f['status']=='PREDICTIONS_FROZEN_BEFORE_TRUTH';assert f['methods']==METHODS+['frozen_a1_a2_k256']
    contract=json.loads(verify(f['contract']).read_text());expected={d+'__'+t for d in ['pile','finance'] for t in contract['evaluation']['targets']}
    assert set(f['cells'])==expected and set(f['predictions'])=={c+'::'+m for c in expected for m in f['methods']}
    for b in f['dependencies']:verify(b)
    obs=json.loads(verify(f['observations']).read_text());selection=json.loads(verify(obs['source_selection']).read_text());pred={}
    for c,cell in f['cells'].items():
        domain=c.split('__')[0];n=contract['evaluation']['sources'][domain];ids=[r['record_id'] for r in selection['records'][domain]]
        assert cell['shape']==[n,128] and cell['record_ids']==ids and len(ids)==len(set(ids))==n and cell['observation']==obs['observations'][c];verify(cell['observation'])
        for method in f['methods']:
            key=c+'::'+method;t=load_file(str(verify(f['predictions'][key])));assert set(t)=={'predictions'};p=t['predictions'];size=64 if method=='frozen_a1_a2_k256' else n
            assert p.dtype==torch.int64 and list(p.shape)==[size,128] and (p[:,0]==128000).all() and (p>=0).all() and (p<128256).all();pred[key]=p
    return f,pred,selection


def score(path,out):
    f,pred,selection=validate(path) # All methods, observations, source order, code, states, predictions BEFORE truth.
    truth={}
    for d,b in selection['curator_payloads'].items():
        t=load_file(str(verify(b)));y=t['token_ids']
        assert y.dtype in [torch.int32,torch.int64] and list(y.shape)==[len(selection['records'][d]),192] and (y[:,0]==128000).all() and t['attention_mask'][:,:128].all()
        assert (y[:,:128]>=0).all() and (y[:,:128]<128256).all()
        truth[d]=y[:,:128].long()
    metrics={};correct={};contrasts={};rng=np.random.default_rng(9013)
    resamples={d:rng.integers(len(y),size=(10000,len(y))) for d,y in truth.items()}
    for key,p in pred.items():
        d=key.split('__')[0];y=truth[d][:len(p)];c=(p[:,1:]==y[:,1:]);correct[key]=c
        metrics[key]={'records':len(p),'tokens':c.numel(),'correct_tokens':int(c.sum()),'token_accuracy':float(c.double().mean()),'residual_errors':int((~c).sum()),'exact_records':int(c.all(1).sum()),'exact_rate':float(c.all(1).double().mean())}
    for cell in f['cells']:
        d=cell.split('__')[0]
        for left,right in [('focused_continuation','ordinary_continuation'),('ordinary_continuation','frozen_B1'),('focused_continuation','frozen_B1'),('frozen_a1_a2_k256','frozen_B1'),('frozen_a1_a2_k256','ordinary_continuation'),('frozen_a1_a2_k256','focused_continuation')]:
            a=correct[cell+'::'+left];b=correct[cell+'::'+right][:len(a)];delta=(a.sum(1)-b.sum(1)).numpy()/127
            ix=resamples[d] if len(a)==len(truth[d]) else np.random.default_rng(9013).integers(len(a),size=(10000,len(a)))
            interval=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()
            contrasts[cell+'::'+left+'-minus-'+right]={'records':len(a),'token_delta':float(delta.mean()),'token_interval95':interval,'token_gains':int((a&~b).sum()),'token_regressions':int((~a&b).sum()),'exact':paired_exact_cp(a.all(1).tolist(),b.all(1).tolist())}
    def passes(left,right):
        p=contrasts['pile__public_base::'+left+'-minus-'+right];l=contrasts['pile__public_lora_2601::'+left+'-minus-'+right]
        checks={'pile_exact_point_5pp':p['exact']['point']>=.05,'pile_exact_lower_positive':p['exact']['lower']>0,'pile_residual_errors_nonincreasing_both_targets':p['token_delta']>=0 and l['token_delta']>=0,'lora_pile_exact_nonnegative':l['exact']['point']>=0}
        for t in ['public_base','public_lora_2601']:
            for control in set([right,'frozen_B1']):checks['finance_'+t+'_vs_'+control]=metrics['finance__'+t+'::'+left]['token_accuracy']>=metrics['finance__'+t+'::'+control]['token_accuracy']-.0005
        return {'passes':all(checks.values()),'checks':checks}
    focused=passes('focused_continuation','ordinary_continuation');ordinary=passes('ordinary_continuation','frozen_B1')
    decision='focused sampling improves on equal-budget ordinary learning' if focused['passes'] else ('ordinary continuation explains the gain' if ordinary['passes'] else 'neither demonstrates a useful fresh-data improvement')
    write_json(out,{'task_id':'TRR-0013','status':'SCORED_AFTER_COMPLETE_PREDICTION_FREEZE','freeze':artifact(path),'metrics':metrics,'contrasts':contrasts,'decision':decision,'decision_checks':{'focused':focused,'ordinary':ordinary},'independent_sources':768,'source_target_observations':1536,'anchor_independent_sources':128,'truth_opened_utc':utc(),'provenance':provenance()});print(json.dumps({'decision':decision,'checks':focused,'metrics':metrics}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['package','predict','anchor','freeze','score','validate']);p.add_argument('--output',required=True);p.add_argument('--package');p.add_argument('--observations');p.add_argument('--standalone');p.add_argument('--anchor');p.add_argument('--freeze');a=p.parse_args();torch.set_num_threads(1)
    if a.command=='package':package(a.output)
    elif a.command in ['predict','anchor']:predict(a.package,a.observations,a.output,a.command=='anchor')
    elif a.command=='freeze':freeze(a.package,a.observations,a.standalone,a.anchor,a.output)
    elif a.command=='score':score(a.freeze,a.output)
    else:validate(a.freeze)
if __name__=='__main__':main()
