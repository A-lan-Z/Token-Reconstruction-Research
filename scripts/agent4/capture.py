"""Evaluator-only public matched capture. Reconstructor never imports this file."""
from common import *
from safetensors.torch import save_file
from transformers import AutoTokenizer
import argparse
p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16');args=p.parse_args()
torch.set_num_threads(2)
source=ROOT/'experiments/agent4-prefix-only-inversion'/f'{args.panel}_sources.json'
records=json.loads(source.read_text());dst=OUT/args.panel/args.dtype;dst.mkdir(parents=True,exist_ok=False)
start=time.perf_counter();prefix=load_prefix(getattr(torch,args.dtype));tokenizer=AutoTokenizer.from_pretrained(OUT/'backup',local_files_only=True)
loaded=time.perf_counter(); tensors={};truth={};metadata=[]
for row in records:
    guard()
    ids=tokenizer.encode(row['text'],add_special_tokens=False)[:row.get('positions',16)-1]
    ids=[128000]+ids
    with torch.no_grad():obs=prefix.forward_full(torch.tensor([ids],device='cuda'))[0].cpu().contiguous()
    tensors[row['id']]=obs;truth[row['id']]=ids
    metadata.append({'id':row['id'],'group':row['group'],'positions':len(ids),'source':row['source']})
save_file(tensors,str(dst/'observations.safetensors'))
write(dst/'evaluator_truth.json',truth)
write(dst/'metadata.json',metadata)
write(EVID/f'{args.panel}_{args.dtype}_capture.json',{'environment':environment(),'source_sha256':digest(source),'observation_sha256':digest(dst/'observations.safetensors'),'truth_sha256':digest(dst/'evaluator_truth.json'),'metadata':metadata,'load_seconds':loaded-start,'capture_seconds':time.perf_counter()-loaded,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
print('Captured',len(records),'records')
