"""Evaluator-only development/final preparation; never imported by predictors."""
from shared import *
from transformers import AutoTokenizer
from safetensors.torch import save_file
import argparse,random,re
p=argparse.ArgumentParser();p.add_argument('--panel',choices=['development','final'],required=True);a=p.parse_args()
torch.set_num_threads(2)
if a.panel=='development':
    texts=[('ordinary','After the rain stopped, the librarian opened the windows and arranged the returned books on a wooden table.'),
           ('ordinary','A small research team checked the measurements twice before sending their report to the village council.'),
           ('stress','cache_7d29af::delta[0x81bc] != key_Zq9; checksum = e3f0a719;')]
    rows=[{'id':f'dev_{i}','group':g,'text':t,'positions':16,'source':'task-generated public development prose/identifier'} for i,(g,t) in enumerate(texts)]
else:
    assert (X/'freeze.json').exists()
    rng=random.Random(947201)
    old=json.loads((ROOT/'experiments/agent4-prefix-only-inversion/final_sources.json').read_text())
    excluded={hashlib.sha256(v['text'].encode()).hexdigest() for v in old}
    rows=[]
    for book in [11,84]:
        data=(OLD/'backup'/f'gutenberg-{book}.txt').read_bytes()
        paras=[t.strip() for t in re.split(r'\n\s*\n',data.decode('utf-8-sig')) if len(t)>300]
        eligible=[i for i in range(len(paras)//4,3*len(paras)//4) if hashlib.sha256(paras[i].encode()).hexdigest() not in excluded]
        for j,i in enumerate(rng.sample(eligible,2)):
            rows.append({'id':f'natural_{book}_{j}','group':'ordinary','text':paras[i],'positions':24,'source':f'https://www.gutenberg.org/ebooks/{book}.txt.utf-8','source_bytes_sha256':hashlib.sha256(data).hexdigest(),'paragraph_index':i})
    for j in range(2):
        value=''.join(rng.choice('abcdef0123456789') for _ in range(36))
        rows.append({'id':f'unusual_{j}','group':'stress','text':f'parse_{value}::lambda_{rng.randrange(10000)}[0x{value[:8]}] != null;','positions':24,'source':'task-generated seed947201'})
    write(E/'source_selection.json',{'seed':947201,'freeze_sha256':digest(X/'freeze.json'),'scope':'unused task-local selections; no canonical or repository-wide blind claim','excluded_original_source_hashes':sorted(excluded)})
write(X/(a.panel+'_sources.json'),rows)
dst=OUT/a.panel;dst.mkdir(exist_ok=False)
tok=AutoTokenizer.from_pretrained(OLD/'backup',local_files_only=True)
prefix=load_prefix(torch.bfloat16);guard();t=time.perf_counter();tensors={};truth={};metadata=[]
for r in rows:
    ids=[128000]+tok.encode(r['text'],add_special_tokens=False)[:r['positions']-1]
    with torch.no_grad():tensors[r['id']]=prefix.forward_full(torch.tensor([ids],device='cuda'))[0].cpu().contiguous()
    truth[r['id']]=ids;metadata.append({k:r[k] for k in ['id','group','source']})
save_file(tensors,str(dst/'observations.safetensors'));write(dst/'evaluator_truth.json',truth);write(dst/'metadata.json',metadata)
write(E/(a.panel+'_capture.json'),{'environment':environment(),'capture_seconds':time.perf_counter()-t,'source_sha256':digest(X/(a.panel+'_sources.json')),'observations_sha256':digest(dst/'observations.safetensors'),'truth_sha256':digest(dst/'evaluator_truth.json'),'records':len(rows),'positions':[len(v) for v in truth.values()]})
print('Captured',a.panel,len(rows),'records without displaying source tokens')
