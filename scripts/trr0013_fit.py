"""Two fixed-readout continuations using the unchanged native step and sampler."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import argparse,json,time,math,os
from dataclasses import asdict
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from scripts.trr0013 import artifact,verify,write_json,save,Guard,provenance,load_start,bank_parts,normalize_payload,batch,native,FixedPublicReadoutHook,load_p09_fixed_state,utc
from scripts.trr0013_vendor.fixed_control_caller import inherited_schedule_steps


def parts(inputs,correction):
    yield from bank_parts(inputs)
    m=json.loads(Path(correction).read_text())
    origin=12000
    for d in ['pile','finance']:
        b=m['observations'][d+'__public_base'];yield origin,b
        origin+=1024


def cache(inputs,correction,guard):
    start=time.perf_counter();t=None;checked=[];fixtures=[]
    for origin,b in parts(inputs,correction):
        raw=normalize_payload(load_file(str(verify(b))))
        if t is None:t={k:torch.empty((14048,*v.shape[1:]),dtype=v.dtype) for k,v in raw.items()}
        n=len(raw['token_ids'])
        for k,v in raw.items():t[k][origin:origin+n].copy_(v)
        for j in [0,n-1]:
            assert all(torch.equal(t[k][origin+j],raw[k][j]) for k in t)
            fixtures.append(origin+j)
        checked.append(b);del raw;guard.check()
    assert origin+n==14048
    return t,{'seconds':time.perf_counter()-start,'verified_payloads':checked,'exact_cache_rows':fixtures,'resources':guard.check()}


def score_rows(model,E,t):
    losses=torch.full(t['token_ids'].shape,float('nan'));correct=torch.zeros_like(t['attention_mask'],dtype=torch.bool)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(losses),8):
            H=t['activations'][start:start+8].to('cuda',dtype=torch.float32);M=t['attention_mask'][start:start+8].to('cuda',dtype=torch.bool)
            Y=t['token_ids'][start:start+8].to('cuda',dtype=torch.long)
            z=model.projected_hidden(H,M);v=M.clone();v[:,0]=False
            for ix in v.nonzero().split(256):
                logits=model.logits_from_rows(z,ix[:,0],ix[:,1],E);y=Y[ix[:,0],ix[:,1]]
                r=start+ix[:,0].cpu();p=ix[:,1].cpu()
                losses[r,p]=F.cross_entropy(logits,y,reduction='none').cpu();correct[r,p]=(logits.argmax(1)==y).cpu()
    return losses,correct


def prepare(correction,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);guard=Guard();clock=time.perf_counter()
    inputs=json.loads((ROOT/'experiments/TRR-0013/inputs.json').read_text());model,E=load_start(inputs)
    old=json.loads((ROOT/'outputs/TRR-0013/original_bank_difficulty_r1/receipt.json').read_text())
    prior=load_file(str(verify(old['difficulty'])))
    m=json.loads(Path(correction).read_text());items=[]
    for d in ['pile','finance']:
        t=normalize_payload(load_file(str(verify(m['observations'][d+'__public_base']))));L,C=score_rows(model,E,t)
        v=t['attention_mask'].clone();v[:,0]=False
        items.append({'losses':L,'correct':C,'valid':v,'labels':t['token_ids'].long()});guard.check()
    combined={k:torch.cat([prior[k]]+[x[k] for x in items]) for k in ['losses','correct','valid','labels']}
    combined['frequency']=torch.bincount(combined['labels'][combined['valid']],minlength=128256)
    b=save(out/'difficulty.safetensors',combined)
    newv=combined['valid'][12000:];newc=combined['correct'][12000:];newL=combined['losses'][12000:][newv]
    schedule={name:{'rows':[],'draws':[],'replacement':[]} for name in ['ordinary','focused']}
    rng=torch.Generator().manual_seed(5013)
    for step in inherited_schedule_steps(combined['valid'],tuple(range(14048)),steps=6000,seed=5013):
        rows=torch.tensor(step.batch_global_rows);draw=torch.tensor(list(zip(step.draw_record_slots,step.draw_position_slots)))
        eligible=combined['valid'][rows].nonzero();values=combined['losses'][rows][eligible[:,0],eligible[:,1]]
        difficult=torch.argsort(values,descending=True,stable=True)[:math.ceil(len(values)/5)]
        picks=difficult[torch.randint(len(difficult),(256,),generator=rng)]
        hard=torch.cat([draw[:256],eligible[picks]])
        for name,d in [('ordinary',draw),('focused',hard)]:
            schedule[name]['rows'].append(rows);schedule[name]['draws'].append(d)
            schedule[name]['replacement'].append(len(torch.unique(d,dim=0))<512)
    sched={};exposure={}
    for name,fields in schedule.items():
        st={'rows':torch.stack(fields['rows']),'draws':torch.stack(fields['draws']),'replacement':torch.tensor(fields['replacement'])}
        sched[name]=save(out/(name+'_schedule.safetensors'),st)
        rr=st['rows'].gather(1,st['draws'][:,:,0]);pp=st['draws'][:,:,1];counts=torch.bincount((rr*192+pp).flatten(),minlength=14048*192)
        frequency=combined['frequency'][combined['labels'][rr,pp]]
        exposure[name]={'draws':rr.numel(),'unique_positions':int((counts>0).sum()),'repeated_draws':int((counts-1).clamp_min(0).sum()),'maximum_repetition':int(counts.max()),'unique_records':int(torch.unique(rr).numel()),'correction_draws':int((rr>=12000).sum()),'actual_support_draws':{'1_5':int((frequency<=5).sum()),'6_50':int(((frequency>5)&(frequency<=50)).sum()),'51_plus':int((frequency>50).sum())}}
    assert torch.equal(torch.stack(schedule['ordinary']['rows']),torch.stack(schedule['focused']['rows']))
    write_json(out/'manifest.json',{'task_id':'TRR-0013','difficulty':b,'schedules':sched,'exposures':exposure,'original_difficulty_reused':old['difficulty'],'correction_manifest':artifact(correction),'contract':artifact(ROOT/'experiments/TRR-0013/contract.json'),'new_pool':{'positions':int(newv.sum()),'errors':int((~newc&newv).sum()),'mean_CE':float(newL.mean()),'p99_CE':float(torch.quantile(newL,.99))},'elapsed_seconds':time.perf_counter()-clock,'resources':guard.check(),'source':artifact(__file__),'provenance':provenance()})
    print(json.dumps({'new_pool_errors':int((~newc&newv).sum()),'elapsed':time.perf_counter()-clock,'exposures':exposure}),flush=True)


def validation(inputs):
    m=json.loads(verify(inputs['validation']['manifest']).read_text());result={}
    for d in ['Pile','Finance']:
        t=load_file(str(verify(m['payloads'][d]['file'])));h=load_file(str(verify(m['source_bindings'][d]['observation_h'])))
        assert torch.equal(t['attention_mask'].bool(),h['attention_mask'].bool()) and torch.equal(t['position_ids'],h['position_ids'])
        result[d]={**t,'activations':h['activations']}
    return result


def evaluate(model,E,val,frequency):
    answer={}
    for d,t in val.items():
        L,C=score_rows(model,E,t);v=t['attention_mask'].bool().clone();v[:,0]=False
        f=frequency[t['token_ids'].long()];bins={}
        for name,m in [('unseen',f==0),('1_5',(f>0)&(f<=5)),('6_50',(f>5)&(f<=50)),('51_plus',f>50)]:
            q=m&v;bins[name]={'positions':int(q.sum()),'errors':int((~C&q).sum())}
        answer[d]={'positions':int(v.sum()),'correct_tokens':int((C&v).sum()),'accuracy':float(C[v].float().mean()),'exact_records':int((C|~v).all(1).sum()),'records':len(v),'mean_CE':float(L[v].mean()),'support':bins}
    return answer


def write_checkpoint(out,model,hook,opt,scheduler,step,inputs,prep,schedule):
    metadata={'schema':'token-reconstruction.trr-p09-fixed-state.v1','method_id':'continued_fixed_readout','selected_step':str(step),'bank_manifest_sha256':prep['sha256'],'fit_manifest_sha256':prep['sha256'],'schedule_semantic_sha256':schedule['sha256'],'base_state_sha256':inputs['starting_state']['sha256'],'embedding_sha256':inputs['readout']['sha256'],'runner_state_sha256':native.method_state_digest(model,hook),'optimizer_state_external':'true','serialization_only':'true'}
    b=save(out/f'step{step}.safetensors',model.state_dict(),metadata)
    optpath=out/f'step{step}.optimizer.pt';tmp=optpath.with_suffix('.partial');torch.save({'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'step':step},tmp);os.replace(tmp,optpath)
    kwargs={'expected_'+k:(int(v) if k=='selected_step' else v) for k,v in metadata.items() if k in ['selected_step','bank_manifest_sha256','fit_manifest_sha256','schedule_semantic_sha256','base_state_sha256','embedding_sha256','runner_state_sha256']};kwargs['expected_state_sha256']=b['sha256']
    restored=load_p09_fixed_state(Path(b['path']),**kwargs)
    assert all(torch.equal(v.detach().cpu(),restored.state_dict()[k]) for k,v in model.state_dict().items())
    del restored
    return {'state':b,'loader_kwargs':kwargs,'optimizer':artifact(optpath),'restore_all_tensor_bytes_equal':True}


def fit(preparation,out,arm,qualify=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);clock=time.perf_counter();guard=Guard();started=utc()
    inputs=json.loads((ROOT/'experiments/TRR-0013/inputs.json').read_text());prep=json.loads(Path(preparation).read_text());verify(prep['contract']);t,cache_evidence=cache(inputs,verify(prep['correction_manifest']),guard)
    model,E=load_start(inputs);hook=FixedPublicReadoutHook('continued_fixed_readout',inputs['readout']['sha256']);opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0,foreach=False);lr=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=6000)
    st=load_file(str(verify(prep['schedules'][arm])));difficulty=load_file(str(verify(prep['difficulty'])));val=validation(inputs)
    source=SimpleNamespace(batch_for_global_rows=lambda rows:batch(t,rows));cfg=native.RunnerConfig(6000,8,512,1000,'prospective_pile_exact_guarded',5013,5e-5,0.,1.,192,2048,'torch.bfloat16')
    curves=[];checkpoints={};train_seconds=0.;log=out/'updates.jsonl'
    def grid(step):
        begin=time.perf_counter();result=evaluate(model,E,val,difficulty['frequency']);curves.append({'step':step,'validation':result,'validation_seconds':time.perf_counter()-begin})
        checkpoints[str(step)]=write_checkpoint(out,model,hook,opt,lr,step,inputs,artifact(preparation),prep['schedules'][arm]);print(json.dumps({'step':step,'validation':result}),flush=True)
    if not qualify:grid(0)
    nsteps=20 if qualify else 6000
    for i in range(nsteps):
        draw=st['draws'][i];step=native.ScheduleStep(i,tuple(st['rows'][i].tolist()),tuple(draw[:,0].tolist()),tuple(draw[:,1].tolist()),bool(st['replacement'][i]))
        r=native.train_one_step(model,hook,source,step,optimizer=opt,embedding=E,config=cfg,activation_dtype=torch.bfloat16);lr.step();train_seconds+=r.elapsed_seconds
        with log.open('a') as f:f.write(json.dumps(asdict(r))+'\n')
        if not qualify and i+1 in [1000,2000,4000,6000]:grid(i+1)
        guard.check()
        if (i+1)%250==0:print(json.dumps({'arm':arm,'step':i+1,'elapsed':time.perf_counter()-clock,'train_seconds':train_seconds}),flush=True)
    if qualify:
        receipt={'status':'QUALIFIED_DISCARDED_20_STEP_RUN','projected_update_seconds_6000':train_seconds/20*6000,'cache':cache_evidence,'resources':guard.check(),'native_geometry':[8,192,2048],'source':artifact(__file__),'provenance':provenance()}
        assert receipt['projected_update_seconds_6000']<1800
    else:
        base=curves[0]['validation']['Finance']['accuracy'];eligible=[c for c in curves if c['validation']['Finance']['accuracy']>=base-.0005]
        def key(c):
            v=c['validation'];return (v['Pile']['exact_records'],(v['Pile']['accuracy']+v['Finance']['accuracy'])/2,-(v['Pile']['mean_CE']+v['Finance']['mean_CE'])/2)
        selected=max(eligible,key=key)
        receipt={'status':'FIT_COMPLETE_SELECTED_BY_PUBLIC_VALIDATION','arm':arm,'selected_step':selected['step'],'selected':checkpoints[str(selected['step'])],'curves':curves,'checkpoints':checkpoints,'cache':cache_evidence,'native_update_seconds':train_seconds,'elapsed_seconds':time.perf_counter()-clock,'preparation':artifact(preparation),'resources':guard.check(),'source':artifact(__file__),'provenance':provenance(),'started_utc':started,'ended_utc':utc()}
    write_json(out/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k in ['status','arm','selected_step','native_update_seconds','elapsed_seconds','resources','projected_update_seconds_6000']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','qualify','fit']);p.add_argument('--correction');p.add_argument('--preparation');p.add_argument('--output',required=True);p.add_argument('--arm',choices=['ordinary','focused'],default='ordinary');a=p.parse_args();torch.set_num_threads(1)
    if a.command=='prepare':prepare(a.correction,a.output)
    else:fit(a.preparation,a.output,a.arm,a.command=='qualify')
if __name__=='__main__':main()
