"""Evaluator-only trusted source selection/capture and post-freeze label rendering."""
from __future__ import annotations
import argparse,importlib,json,sys,time,gc
from pathlib import Path
import torch
from safetensors.torch import save_file
from scripts.agent3_hybrid.common import *
RANGES={'pile':(0,10000),'finance':(50000,52000)}
SEED=5103

def shared():
    for p in (SHARED,SHARED/'src'):sys.path.insert(0,str(p))
    return importlib.import_module('scripts.trr_p12.capture'),importlib.import_module('scripts.trr_p12.select_panel')

def normalize(capture):
    source=SHARED/'experiments/TRR-P11/selector/public_source_inputs_r1.json'
    if binding(source)['sha256']!='1fa6aea4485ad602d396e6a57dc53257977a0f4581375892ef22ab176d52b409':raise ValueError('source input binding changed')
    return capture.p11._normalize_source_inputs(source,root=SHARED,require_tokenizer_dir=True),source

def select(a):
    check_implementation();capture,selector=shared();inputs,source=normalize(capture)
    union,ub=selector.load_union(SHARED/'experiments/TRR-P12/exclusions/identity_union_extension_r2.json')
    exclusions=[ub]
    for name in ('correction_selection_r1','evaluation_selection_r1'):
        b,receipt=selector.load_opaque_reservation(ROOT.parent/f'TRR-0013/outputs/TRR-0013/{name}/opaque_reservations.json');union=selector.merge_identity_bundles(union,b);exclusions.append(receipt)
    oldpanel=SHARED/'outputs/TRR-P12/sources-r2/panel.json';old,oldbinding,_=capture.load_panel(oldpanel,'pile');exclusions.append(binding(oldpanel))
    # Exclude every P12 source, including PR25 development and P12's larger panel.
    for domain,rows in old['selection_rule']['records'].items():
        for row in rows:
            namespace=selector.p10.Namespace(domain,row['dataset_id'],row['split'],row['revision'])
            for k in ('record_id','source_index','h40_sequence_sha256','h128_sequence_sha256','h129_sequence_sha256','trr0002_active_token_ids_sha256','trr0002_h40_token_ids_sha256'):
                if row.get(k) is not None:union.add(k,row[k],namespace)
            union.add('rendered_sha256',row['public_record_sha256'],namespace)
    tokenizer=capture.trusted._load_tokenizer(Path(inputs['tokenizer']['path']))
    from token_reconstruction.trr0005_public_corpus import deterministic_row_order
    selected={};stats={};seen=set()
    for domain in DOMAINS:
        dataset=capture.trusted._load_arrow_dataset(tuple(Path(x['path']) for x in inputs[domain]['arrow_files']))
        selected[domain]=[];counts={'short_or_malformed':0,'excluded':0,'duplicate':0,'visited':0}
        for index in deterministic_row_order(range(*RANGES[domain]),dataset_key='agent3-static-prefix-'+domain,seed=SEED):
            counts['visited']+=1
            try:candidate=capture.trusted._render_row(domain,dataset[index],index,tokenizer);meta=capture.p11._candidate_identity(candidate)
            except Exception as e:
                if any(x in str(e).lower() for x in ('shorter than','no user/assistant','malformed')):counts['short_or_malformed']+=1;continue
                raise
            if capture.p11._candidate_exclusion_reasons(meta,union):counts['excluded']+=1;continue
            keys={meta['record_id'],meta['public_record_sha256'],meta['h128_sequence_sha256']}
            if seen&keys:counts['duplicate']+=1;continue
            seen|=keys;selected[domain].append(capture.p11._selection_row(candidate))
            if len(selected[domain])==32:break
        if len(selected[domain])!=32:raise ValueError('insufficient fresh sources: '+domain)
        stats[domain]=counts;del dataset
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    panel={'schema':'agent3-static-hybrid-panel-v1','status':'FROZEN_FRESH_PANEL_NO_RECONSTRUCTION_OUTCOMES','created_utc':utc(),'code_commit':commit(),'implementation':binding(EV/'implementation-freeze.json'),'source_inputs':binding(source),'shared_code':capture.code_bindings(),'exclusions':exclusions,'seed':SEED,'ranges':RANGES,'records':selected,'selection_statistics':stats,'truth_emitted':False,'all_p12_sources_excluded':True,'p03_accessed':False}
    write_json(out/'panel.json',panel);print(json.dumps({'status':panel['status'],'panel':binding(out/'panel.json'),'counts':{d:len(v) for d,v in selected.items()}}))

def rendered(panel,domain,capture):
    for b in panel['shared_code'].values():verify(b)
    inputs,_=normalize(capture);tokenizer=capture.trusted._load_tokenizer(Path(inputs['tokenizer']['path']))
    dataset=capture.trusted._load_arrow_dataset(tuple(Path(x['path']) for x in inputs[domain]['arrow_files']))
    sequences=[]
    for row in panel['records'][domain]:
        candidate=capture.trusted._render_row(domain,dataset[row['row_index']],row['row_index'],tokenizer)
        actual=capture.p11._candidate_identity(candidate)
        if any(str(actual.get(k))!=str(row.get(k)) for k in capture.ROW_FIELDS):raise ValueError('source identity changed')
        sequences.append(list(candidate.token_ids[:128]))
    return sequences,tokenizer

def capture_cell(a):
    check_implementation();capture,_=shared();panel=json.loads(Path(a.panel).read_text());verify(panel['implementation'])
    sequences,_=rendered(panel,a.domain,capture)
    batch=capture.pad_public_token_sequences(sequences,maximum_tokens=192,pad_token_id=128001,bos_token_id=128000,vocab_size=128256)
    trajectory=SHARED/'outputs/TRR-P12/target-r1/run.json'
    if binding(trajectory)['sha256']!='d0d04c971b438df1bf0580bd181fbf574942e5e4e0dfe2901b0fb535ef84ad66':raise ValueError('trajectory changed')
    run=json.loads(trajectory.read_text());adapter=None
    if a.stage:
        record=next(x for x in run['stage_records'] if x['stage_updates']==a.stage);adapter=verify(record['adapter']);verify(record['backup']['backup'])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);device=torch.device('cuda');start=time.monotonic()
    prefix=capture.qualified.load_prefix(capture.DEFAULT_MODEL,device=device,adapter=adapter)
    def guard():capture.qualified.guard_resources(device,phase='agent3_capture',started=start,timeout_seconds=600,min_free_gpu_bytes=3*2**30,max_reserved_gpu_bytes=8*2**30,max_rss_bytes=12*2**30)
    h=capture.capture_public_prefix(prefix,batch,device=device,batch_size=8,hidden_size=2048,resource_check=guard)
    artifact=out/'observation.safetensors';save_file({'activations':h[:,:128].cpu().contiguous(),'attention_mask':batch.attention_mask[:,:128].byte().contiguous(),'position_ids':batch.position_ids[:,:128].long().contiguous()},str(artifact))
    contract={'schema':'agent3-static-hybrid-observation-v1','domain':a.domain,'stage':a.stage,'record_ids':[r['record_id'] for r in panel['records'][a.domain]],'observations':binding(artifact)}
    write_json(out/'contract.json',contract)
    write_json(out/'capture.json',{'status':'COMPLETE_SANITIZED_OBSERVATION','code_commit':commit(),'panel':binding(a.panel),'trajectory':binding(trajectory),'adapter':None if adapter is None else binding(adapter),'contract':binding(out/'contract.json'),'elapsed_seconds':time.monotonic()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'source_token_ids_emitted':False,'profile':'P12 full-sequence cut4 BF16 SDPA B8x192,128 active; only H128 released'})
    print(json.dumps({'status':'CAPTURE_COMPLETE','domain':a.domain,'stage':a.stage,'seconds':time.monotonic()-start}))

def truth(a):
    # Import the evaluator gate only after the caller explicitly requests the truth phase.
    from scripts.agent3_hybrid.evaluate import load_freeze
    frozen=load_freeze(Path(a.freeze));check_implementation();gate_time=utc()
    panel=json.loads(verify(frozen['panel']).read_text());capture,_=shared();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'EVALUATOR_TRUTH_AFTER_COMPLETE_FREEZE','freeze':binding(a.freeze),'freeze_checked_utc':gate_time,'domains':{},'code_commit':commit()}
    for domain in DOMAINS:
        sequences,_=rendered(panel,domain,capture);artifact=out/f'{domain}.safetensors';save_file({'token_ids':torch.tensor(sequences,dtype=torch.int64)},str(artifact))
        manifest['domains'][domain]={'artifact':binding(artifact),'record_ids':[r['record_id'] for r in panel['records'][domain]]}
    write_json(out/'truth.json',manifest);print(json.dumps({'status':manifest['status'],'manifest':binding(out/'truth.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='phase',required=True)
    s=sub.add_parser('select');s.add_argument('--output',required=True)
    c=sub.add_parser('capture');c.add_argument('--panel',required=True);c.add_argument('--domain',choices=DOMAINS,required=True);c.add_argument('--stage',type=int,choices=STAGES,required=True);c.add_argument('--output',required=True)
    t=sub.add_parser('truth');t.add_argument('--freeze',required=True);t.add_argument('--output',required=True)
    a=p.parse_args();{'select':select,'capture':capture_cell,'truth':truth}[a.phase](a)
