"""Trusted source/capture process. Never imported by the reconstruction process."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import argparse,json,hashlib,time,datetime
import torch
from safetensors.torch import load_file
from scripts.trr0013 import artifact,verify,write_json,save,Guard,provenance
from scripts import trr0005_produce_confirmation as trusted
from scripts.trr_p11 import source_selector as selector
from scripts.trr_p10.build_exclusion_audit import IdentityBundle,Namespace
from token_reconstruction.trr0005_public_corpus import deterministic_row_order
from token_reconstruction.public_activation import pad_public_token_sequences
from scripts.trr0011_capture import capture_full_forward,load_public_target_model

SNAPSHOT=Path('/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6')
LORA=ROOT.parent.parent/'outputs/TRR-0002/public-calibration/updates/public_lora_2601.safetensors'


def union_load(bindings):
    union=IdentityBundle('trr0013_union','union',Path('.'),'0'*64,0)
    for binding in bindings:
        x=json.loads(verify(binding).read_text())
        fields=x['fields']
        for field,namespaces in fields.items():
            if field not in selector._P11_UNION_FIELDS:raise ValueError(field)
            for namespace,values in namespaces.items():
                ns=Namespace(*namespace.split('|'))
                for value in values:union.add(field,value,ns)
    return union


def add_metadata(union,m):
    ns=Namespace(m['dataset_key'],m['dataset_id'],m['split'],m['revision'])
    for key in ['record_id','h128_sequence_sha256','h129_sequence_sha256','h40_sequence_sha256','trr0002_active_token_ids_sha256','trr0002_h40_token_ids_sha256']:
        if m.get(key):union.add(key,m[key],ns)
    union.add('rendered_sha256',m['public_record_sha256'],ns)
    union.add('source_index',int(m['source_index']),ns)


def select(contract_path,exclusions_path,phase,output,additional=None):
    started=time.perf_counter();contract=json.loads(Path(contract_path).read_text());bindings=json.loads(Path(exclusions_path).read_text())['bindings']
    union=union_load(bindings)
    if additional:
        prior=json.loads(Path(additional).read_text())
        for rows in prior['records'].values():
            for m in rows:add_metadata(union,m)
    inputs=json.loads((ROOT/'experiments/TRR-0013/inputs.json').read_text())
    vm=json.loads(verify(inputs['validation']['manifest']).read_text())
    tokenizer=trusted._load_tokenizer(SNAPSHOT)
    sources={d:[verify(b) for b in vm['source_bindings'][d.title()]['dataset_descriptor']['arrow_files']] for d in ['pile','finance']}
    datasets={d:trusted._load_arrow_dataset(paths) for d,paths in sources.items()}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rule=contract['public_pool'] if phase=='correction' else contract['evaluation']
    quotas=rule['correction_records'] if phase=='correction' else rule['sources']
    ranges=rule['pool_ranges'] if phase=='correction' else rule['ranges']
    records={};payloads={};counts={};opaque_fields={}
    for domain in ['pile','finance']:
        chosen=[];ids=[];rejected={};begin,end=ranges[domain]
        order=deterministic_row_order(range(begin,end),dataset_key=f'trr0013-{phase}-{domain}',seed=rule['selection_seed'])
        for i in order:
            if any(a <= i < b for a,b in contract.get('temporary_source_reservations',{}).get(domain,[])):
                rejected['parallel_reserved']=rejected.get('parallel_reserved',0)+1;continue
            try:
                candidate=trusted._render_row(domain,datasets[domain][i],i,tokenizer)
                m=selector._candidate_identity(candidate)
            except trusted.ProducerError as e:
                text=str(e).lower()
                if not any(s in text for s in ['shorter than','no user/assistant','malformed']):raise
                rejected['invalid']=rejected.get('invalid',0)+1;continue
            reasons=selector._candidate_exclusion_reasons(m,union)
            if reasons:
                rejected['excluded']=rejected.get('excluded',0)+1;continue
            chosen.append(m);ids.append(list(candidate.token_ids[:192]));add_metadata(union,m)
            if len(chosen)==quotas[domain]:break
        if len(chosen)!=quotas[domain]:
            write_json(out/'insufficient_capacity.json',{'domain':domain,'eligible_found':len(chosen),'required':quotas[domain],'range':ranges[domain],'rejected':rejected,'status':'FAILED_NO_COMPLETED_PANEL'})
            raise RuntimeError('Insufficient eligible sources; freeze extension before retry')
        padded=pad_public_token_sequences(ids,maximum_tokens=192,bos_token_id=128000,pad_token_id=128001)
        payloads[domain]=save(out/f'{domain}_curator_tokens.safetensors',{'token_ids':padded.token_ids,'attention_mask':padded.attention_mask,'position_ids':padded.position_ids})
        records[domain]=chosen;counts[domain]={'selected':len(chosen),'rejected':rejected}
        for m in chosen:
            ns='|'.join(m[k] for k in ['dataset_key','dataset_id','split','revision'])
            for f,k in [('source_index','source_index'),('record_id','record_id'),('rendered_sha256','public_record_sha256'),('h128_sequence_sha256','h128_sequence_sha256'),('h129_sequence_sha256','h129_sequence_sha256'),('h40_sequence_sha256','h40_sequence_sha256')]:
                if m.get(k) is not None:opaque_fields.setdefault(f,{}).setdefault(ns,[]).append(m[k])
    write_json(out/'selection.json',{'task_id':'TRR-0013','phase':phase,'records':records,'curator_payloads':payloads,'counts':counts,'contract':artifact(contract_path),'exclusions':bindings,'public_source_files':{d:[artifact(p) for p in ps] for d,ps in sources.items()},'tokenizer':trusted._tokenizer_descriptor(SNAPSHOT),'private_source_tokens':'Curator/capture only until prediction freeze; reconstruction receives observations only.','elapsed_seconds':time.perf_counter()-started,'curator_source':artifact(__file__),'provenance':provenance()})
    write_json(out/'opaque_reservations.json',{'task_id':'TRR-0013','phase':phase,'fields':{f:{ns:sorted(set(v)) for ns,v in n.items()} for f,n in opaque_fields.items()},'source_text_or_tokens':False,'counts':counts})
    print(json.dumps({'selection':artifact(out/'selection.json'),'opaque_reservations':artifact(out/'opaque_reservations.json'),'counts':counts}),flush=True)


def capture(selection_path,output,phase):
    s=json.loads(Path(selection_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();guard=Guard(seconds=1200,rss_gib=16)
    targets=['public_base'] if phase=='correction' else ['public_base','public_lora_2601']
    result={};qualifiers={}
    for target in targets:
        role='historical_trained_benchmark_adaptation' if target!='public_base' else 'public_base'
        model=load_public_target_model(SNAPSHOT,variant={'role':role},device=torch.device('cuda'),historical_lora_path=LORA if target!='public_base' else None)
        for domain in ['pile','finance']:
            t=load_file(str(verify(s['curator_payloads'][domain])))
            batch=pad_public_token_sequences([row[:int(mask.sum())].tolist() for row,mask in zip(t['token_ids'],t['attention_mask'])],maximum_tokens=192,bos_token_id=128000,pad_token_id=128001)
            assert all(torch.equal(getattr(batch,k),t[k]) for k in t)
            small=pad_public_token_sequences([row[:int(mask.sum())].tolist() for row,mask in zip(t['token_ids'][:8],t['attention_mask'][:8])],maximum_tokens=192,bos_token_id=128000,pad_token_id=128001)
            q=capture_full_forward(model,small,device=torch.device('cuda'),batch_size=8,resource_check=guard.check)
            qualifiers[f'{domain}__{target}']={'resources':guard.check(),'batch_geometry':[8,192,2048]}
            write_json(out/f'{domain}__{target}_qualification.json',qualifiers[f'{domain}__{target}'])
            H=capture_full_forward(model,batch,device=torch.device('cuda'),batch_size=8,resource_check=guard.check)
            assert torch.equal(q,H[:8]),'Capture qualification output changed'
            qualifiers[f'{domain}__{target}']['output_exact']=True
            del q,small
            if phase=='correction':
                values={'activations':H,**t}
            else:
                values={'activations':H[:,:128].contiguous(),'attention_mask':t['attention_mask'][:,:128].contiguous(),'position_ids':t['position_ids'][:,:128].contiguous()}
            result[f'{domain}__{target}']=save(out/f'{domain}__{target}.safetensors',values)
            del H,t,batch;guard.check()
        del model;torch.cuda.empty_cache()
    write_json(out/'manifest.json',{'task_id':'TRR-0013','phase':phase,'observations':result,'qualifiers':qualifiers,'source_selection':artifact(selection_path),'contract':s['contract'],'public_model':artifact(SNAPSHOT/'model.safetensors'),'lora':artifact(LORA) if phase!='correction' else None,'elapsed_seconds':time.perf_counter()-started,'resources':guard.check(),'reconstruction_truth_access':False,'source_tokens_only_in_correction_fitting_payloads':phase=='correction','curator_source':artifact(__file__),'provenance':provenance()})
    print(json.dumps({'manifest':artifact(out/'manifest.json'),'elapsed_seconds':time.perf_counter()-started}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','capture']);p.add_argument('--phase',choices=['correction','evaluation'],required=True);p.add_argument('--output',required=True);p.add_argument('--contract',default='experiments/TRR-0013/contract.json');p.add_argument('--exclusions',default='experiments/TRR-0013/exclusions.json');p.add_argument('--selection');p.add_argument('--additional')
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='select':select(a.contract,a.exclusions,a.phase,a.output,a.additional)
    else:capture(a.selection,a.output,a.phase)

if __name__=='__main__':main()
