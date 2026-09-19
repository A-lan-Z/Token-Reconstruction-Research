"""Evaluator-only source selection and paired capture; not imported by predictor."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/trr0014"))
from native import *
from transformers import AutoTokenizer
from token_reconstruction.target_update import TargetLoRAConfig,install_target_lora,load_target_lora
import random,re,types
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
X=ROOT/'experiments/TRR-0016';OUT=ROOT/'outputs/TRR-0016'
assert (X/'registry.json').exists()
dst=OUT/'fresh_r3';dst.mkdir(exist_ok=True)
assert not any(dst.iterdir()), 'refuse to overwrite capture artifacts'
rng=random.Random(1616256)
tok=AutoTokenizer.from_pretrained(ASSETS/'backup',local_files_only=True)
excluded=set()
for file in [ROOT/'experiments/agent4-prefix-only-inversion/final_sources.json',SIB/'agent4-prefix-only-rescue/experiments/agent4-prefix-only-rescue/final_sources.json',SIB/'agent4-prefix-only-rescue/experiments/agent4-prefix-only-rescue/development_sources.json']:
    for row in json.loads(file.read_text()):excluded.add(hashlib.sha256(row['text'].encode()).hexdigest())
for row in json.loads((SIB/'TRR-0014/outputs/TRR-0014/fresh_r1/sources.json').read_text())+json.loads((SIB/'TRR-0014/outputs/TRR-0014/fresh_r2/sources.json').read_text()):
    if 'source_sha256' in row:excluded.add(row['source_sha256'])
rows=[];availability={}
for book in [11,84]:
    f=ASSETS/'backup'/('gutenberg-'+str(book)+'.txt')
    paras=[t.strip() for t in re.split(r'\n\s*\n',f.read_text(encoding='utf-8-sig')) if len(t)>300]
    eligible=[]
    for i in range(len(paras)//10,9*len(paras)//10):
        if hashlib.sha256(paras[i].encode()).hexdigest() in excluded:continue
        ids=tok.encode(paras[i],add_special_tokens=False)
        if len(ids)>=127:eligible.append((i,ids[:127]))
    availability[str(book)]=len(eligible)
    for j,(i,ids) in enumerate(rng.sample(eligible,16)):
        rows.append({'source_id':f'book{book}_{j:02d}','group':'natural','tokens':[128000]+ids,'source_file_sha256':digest(f),'paragraph_index':i,'source_sha256':hashlib.sha256(paras[i].encode()).hexdigest()})
for j in range(8):
    value=''.join(rng.choice('abcdef0123456789') for _ in range(80))
    text=f'parse_{value}::lambda_{rng.randrange(10000)}[0x{value[:8]}] != null; '
    ids=tok.encode(text*3,add_special_tokens=False)[:39]
    assert len(ids)==39
    rows.append({'source_id':f'code_{j:02d}','group':'stress','tokens':[128000]+ids,'source':'generated seed1616256'})
write(dst/'sources.json',rows)
write(X/'fresh_r3_selection.json',{'seed':1616256,'available':availability,'excluded_hashes':sorted(excluded),'sources_sha256':digest(dst/'sources.json'),'plan_sha256':digest(X/'PLAN.md'),'scope':'task-local new paragraph selections'})
prefix=load_prefix();guard();outputs={};truth={};meta=[];times={}
adapter=SIB/'TRR-P12/outputs/TRR-P12/target-r1/target-lora-stage-0256.safetensors'
assert digest(adapter)=='a6f13af1835743556ca80c8fed133832fb010bea2957014499740e4b0fe58b0a'
for condition in ['matched','lora256']:
    if condition=='lora256':
        installed=install_target_lora(types.SimpleNamespace(model=prefix),TargetLoRAConfig(rank=8,alpha=16,seed=6012))
        load_target_lora(installed,adapter);prefix.requires_grad_(False)
    sync();start=time.perf_counter()
    for row in rows:
        guard();key=condition+'__'+row['source_id']
        with torch.no_grad():outputs[key]=prefix.forward_full(torch.tensor([row['tokens']],device='cuda'))[0].cpu().contiguous()
        truth[key]=row['tokens'];meta.append({'id':key,'source_id':row['source_id'],'group':row['group'],'condition':condition,'positions':len(row['tokens'])})
    sync();times[condition]=time.perf_counter()-start
save_file(outputs,str(dst/'observations.safetensors'));write(dst/'evaluator_truth.json',truth);write(dst/'metadata.json',meta)
drift={}
for group in ['natural','stress']:
    vals=[]
    for row in rows:
        if row['group']!=group:continue
        a=outputs['matched__'+row['source_id']][1:].float();b=outputs['lora256__'+row['source_id']][1:].float()
        vals.append(float((b-a).norm()/a.norm()))
    drift[group]={'mean_relative_l2':sum(vals)/len(vals),'values':vals}
write(X/'fresh_r3_capture.json',{'environment':environment(),'times':times,'drift':drift,'observations_sha256':digest(dst/'observations.safetensors'),'truth_sha256':digest(dst/'evaluator_truth.json'),'metadata_sha256':digest(dst/'metadata.json'),'adapter_sha256':digest(adapter),'source_count':len(rows),'condition_count':2,'prediction_truth_opened':False,'target_weights':'evaluator only; reconstructor public prefix unchanged','peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
print('Captured',len(meta),'paired observations without revealing labels',flush=True)
